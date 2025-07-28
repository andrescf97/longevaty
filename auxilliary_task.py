import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import Dataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, RandRotated, RandFlipd, ToTensord, Resized
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import wandb
import json
import os
from tvital.eva import Eva, PatchEmbed
from tvital.vit import ViT
from utils.masker import Masker
from einops import rearrange
from vital.models.attention import MultiHeadAttention
from torch.nn import MultiheadAttention
from monai import transforms


class Permuted(transforms.MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        image = data["image"]
        # Permute the image dimensions for 3d conv patch embedding
        permuted_image = image.permute(0, 3, 1, 2)
        data['image'] = permuted_image
        return data

class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()

        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        #self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob
    
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p=0.1):
        """
        A residual block that projects from `dim` to `hidden_dim`, applies a non-linearity
        and dropout, then projects back to `dim`, and adds the original input.
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + identity
    
    
class FusionLayerWithResidual(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=1024, output_dim=768, dropout_p=0.1, num_residual_blocks=2):
        """
        Fuses an input vector of dimension `input_dim` through a multi-layer residual network,
        then projects it down to an output vector of dimension `output_dim`.
        """
        super(FusionLayerWithResidual, self).__init__()
        # Optional initial projection to help stabilize training
        self.fc_in = nn.Linear(input_dim, input_dim)
        # Create a sequence of residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(input_dim, hidden_dim, dropout_p) 
            for _ in range(num_residual_blocks)
        ])
        # Final projection from input_dim (2304) to output_dim (768)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x is assumed to be of shape (batch_size, input_dim)
        x = self.fc_in(x)
        x = self.res_blocks(x)
        x = self.fc_out(x)
        x = self.act(x)
        return x




class Lungevity(nn.Module):
    def __init__(
        self,
        transformer: str = "eva",
        patch_size: int = 16,
        grid_size: list = [10, 10, 10],
        enc_dim: int = 792,
        enc_blocks: int = 12,
        enc_heads: int = 12,
        dropout_rate: float = 0.2,
        num_reg_tokens: int = 0,  # Number of additional tokens for the encoder
        use_cls: bool = True,
        hidden_dim: int = 792,
        max_followup: int = 6,
        fusion_layer: bool = True,
        guided_attention_heads: int = 4,
        use_mean_token: bool = False,
    ):
        super().__init__()
        
        if transformer == "eva":
            self.encoder =  Eva(
                embed_dim=enc_dim,
                depth=enc_blocks,
                num_heads=enc_heads,
                pos_drop_rate=0.0,
                patch_drop_rate=0.0,
                proj_drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                ref_feat_shape=grid_size,
                num_reg_tokens=num_reg_tokens,  # Assuming 1 prefix ("cls") token for the encoder
            )

        else:
            self.encoder = ViT(
                embed_dim=enc_dim,
                grid_size=grid_size,
                depth=enc_blocks,
                num_heads=enc_heads,
                drop_rate=dropout_rate
            )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.down_projection = PatchEmbed(patch_size, input_channels=1, embed_dim=enc_dim)

        hidden = hidden_dim
        self.aggregate_fn = lambda x, y, z: x
        if use_cls:
            hidden += hidden_dim
            self.aggregate_fn = lambda x, y, z: torch.concat([x, y], axis=-1)
        if use_mean_token:
            hidden += hidden_dim
            self.aggregate_fn = lambda x, y, z: torch.concat([x, z], axis=-1)
        if use_cls and use_mean_token:
            self.aggregate_fn = lambda x, y, z: torch.concat([x, y, z], axis=-1)

        self.mha = MultiheadAttention(embed_dim=hidden_dim, num_heads=guided_attention_heads, batch_first=True, dropout=dropout_rate)

        self.classifier = nn.Linear(792, 2)

    def attention_pooling(
            self,
            tokens: torch.Tensor
    ):
        q = tokens[:, 0:1, :]
        k = tokens[:, 1:, :]
        v = tokens[:, 1:, :]

        attns, attn_weights = self.mha(q, k, v)
        return attns.squeeze(axis=1), attn_weights.mean(1).squeeze()

    def __call__(
        self,
        input: torch.Tensor,
    ):
        
        x, FD, FW, FH = self.patch_embed(input) #TODO where is cls
        # add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        embeddings = self.encoder(x)

        cls = embeddings[:, 0, :]
        pred = self.classifier(cls)
        return pred, cls

    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding

        # timm"s trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, "cls_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights) # TODO
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patch_embed(self, x):
        FD, FW, FH = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        B, C, D, W, H = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c d w h -> b (d w h) c")
        return x, FD, FW, FH

    def restore_image(self, x, D, W, H):
        x = rearrange(x, "b (d w h) c -> b c d w h", h=H, w=W, d=D)


    # def forward(self, x):
    #     x, FD, FW, FH = self.patch_embed(x)
    #     x_masked, mask, ids_restore, ids_keep = self.masker(x)
    #     cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
    #     x_masked = torch.cat([cls_token, x_masked], dim=1)

    #     x_masked = self.encoder(x_masked, ids_keep)
    #     x_masked = self.encoding_projection(x_masked)

    #     masked_tokens = self.mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] + 1 - x_masked.shape[1], 1)
    #     all_embeddings = torch.cat((x_masked[:, 1:, :], masked_tokens), dim=1)
    #     all_embeddings = torch.gather(all_embeddings, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_masked.shape[2]))
    #     all_embeddings = torch.cat((x_masked[:, :1, :], all_embeddings), dim=1)

    #     recon_seq = self.decoder(all_embeddings)
    #     recon_seq = self.up_sample(recon_seq)
    #     return recon_seq[:, 1:, :], mask
    

def patchify(im: torch.Tensor, patch_size: list[int, int, int] = [5, 16, 16]):
    """Split image into patches of size patch_size.

    im: [B, S, T, H, W]
    patch_size: a list of 3
    x: [B, L, np.prod(patch_size)] where L = S * T * H * W / np.prod(patch_size)
    """
    assert len(im.shape) == 5
    assert len(patch_size) == 3

    B, S, T, H, W = im.shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = im.reshape(B, S, t, patch_size[0], h, patch_size[1], w, patch_size[2])
    x = torch.einsum("bstphqwr->bsthwpqr", x)
    x = x.reshape(B, S * t * h * w, np.prod(patch_size))
    return x


def unpatchify(x: torch.Tensor, im_shape: list[int], patch_size: list[int, int, int] = [5, 16, 16]):
    """Combine patches into image.

    x: [B, L, np.prod(patch_size) or T * np.prod(patch_size)]
    im_shape: [B, S, T, X, Y]
    im: [B, S, T, X, Y] where X = Y
    """
    assert len(x.shape) == 3
    assert len(patch_size) == 3
    assert len(im_shape) == 5

    B, S, T, H, W = im_shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = x.reshape(B, S, t, h, w, patch_size[0], patch_size[1], patch_size[2])
    x = torch.einsum("bsthwpqr->bstphqwr", x)
    im = x.reshape(im_shape)
    return im







# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

        

def create_transforms(train=True):
    """Create MONAI transforms for training and validation"""
    if train:
        transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(
                keys=["image"],
                mode=["trilinear"],
                spatial_size=[160, 240, 128],
            ),
            Permuted(keys=["image"]),
            ToTensord(keys=["image", 'sex', 'smoking_status', 'pleural_effusion', 'nodule_greater_4mm', 'emphysema', 'fibrosis'])
        ])
    else:
        transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(
                keys=["image"],
                mode=["trilinear"],
                spatial_size=[160, 240, 128],
            ),
            Permuted(keys=["image"]),
            ToTensord(keys=["image", 'sex', 'smoking_status', 'pleural_effusion', 'nodule_greater_4mm', 'emphysema', 'fibrosis'])
        ])
    
    return transforms

def train_step(model, batch, criterion, optimizer, scaler, task_name='smoking_status'):
    """Single training step with mixed precision for CLS tokens"""
    model.train()
    
    cls_tokens = batch['cls'].to(device)
    labels = batch[task_name].to(device).long()  # Convert to long for CrossEntropyLoss
    
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        logits = model(cls_tokens)
        loss = criterion(logits, labels)
    
    # Mixed precision backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Get predictions
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    return loss.item(), preds.detach().cpu().numpy(), labels.detach().cpu().numpy()

def validate_step(model, batch, criterion, task_name='smoking_status'):
    """Single validation step for CLS tokens"""
    model.eval()
    
    cls_tokens = batch['cls'].to(device)
    labels = batch[task_name].to(device).long()
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            logits = model(cls_tokens)
            loss = criterion(logits, labels)
    
    # Get predictions
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    return loss.item(), preds.detach().cpu().numpy(), labels.detach().cpu().numpy()

def compute_metrics(y_true, y_pred):
    """Compute classification metrics"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def load_data_from_json(json_path):
    """Load data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to MONAI format if needed
    monai_data = []
    for item in data:
        monai_item = {
            'image': item['image'],  # Path to image file
            'smoking_status': item['smoking_status']  # 0 for female, 1 for male (or adjust as needed)
        }
        monai_data.append(monai_item)
    
    return monai_data

def extract_cls_tokens(model, data_loader, split_name, checkpoint_name, device):
    """Extract CLS tokens from the model and save them with all tabular features"""
    model.eval()
    cls_tokens = []
    
    # Initialize lists for all tabular features
    all_labels = {
        'sex': [],
        'smoking_status': [],
        'pleural_effusion': [],
        'nodule_greater_4mm': [],
        'emphysema': [],
        'fibrosis': []
    }
    
    print(f"Extracting CLS tokens for {split_name}...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting {split_name} CLS tokens"):
            images = batch['image'].to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                _, cls = model(images)
            
            cls_tokens.append(cls.cpu().numpy())
            
            # Collect all tabular features
            for feature_name in all_labels.keys():
                if feature_name in batch:
                    all_labels[feature_name].append(batch[feature_name].cpu().numpy())
                else:
                    # If feature is missing, add placeholder (you might want to handle this differently)
                    all_labels[feature_name].append(np.full(len(batch['image']), -1))
    
    cls_tokens = np.concatenate(cls_tokens, axis=0)
    
    # Concatenate all labels
    for feature_name in all_labels.keys():
        all_labels[feature_name] = np.concatenate(all_labels[feature_name], axis=0)
    
    # Save the extracted features
    save_dir = '/pool/data/lung/NLST/cls_tokens'
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'{checkpoint_name}_{split_name}.npz')
    
    # Save CLS tokens and all labels
    save_dict = {'cls_tokens': cls_tokens}
    save_dict.update(all_labels)
    
    np.savez(save_path, **save_dict)
    
    print(f"Saved {len(cls_tokens)} CLS tokens with all tabular features to {save_path}")
    return save_path

def load_cls_tokens(checkpoint_name, split_name, task_name='smoking_status'):
    """Load pre-computed CLS tokens and specific task labels"""
    save_dir = '/pool/data/lung/NLST/cls_tokens'
    save_path = os.path.join(save_dir, f'{checkpoint_name}_{split_name}.npz')
    
    if os.path.exists(save_path):
        data = np.load(save_path)
        cls_tokens = data['cls_tokens']
        
        # Load the specific task labels
        if task_name in data:
            labels = data[task_name]
            return cls_tokens, labels
        else:
            print(f"Warning: Task '{task_name}' not found in saved data. Available tasks: {list(data.keys())}")
            return cls_tokens, None
    else:
        return None, None

class CLSDataset(torch.utils.data.Dataset):
    """Dataset for pre-computed CLS tokens"""
    def __init__(self, cls_tokens, labels, task_name='smoking_status'):
        self.cls_tokens = torch.from_numpy(cls_tokens).float()
        self.labels = torch.from_numpy(labels).long()
        self.task_name = task_name
    
    def __len__(self):
        return len(self.cls_tokens)
    
    def __getitem__(self, idx):
        return {
            'cls': self.cls_tokens[idx],
            self.task_name: self.labels[idx]
        }

def main():
    # Configuration
    checkpoint_path = '/pool/data/lung/NLST/checkpoints/mae/silver-armadillo-411.ckpt'
    
    # SELECT TASK HERE - Change this to run different classification tasks
    TASK_NAME = 'smoking_status'  # Options: 'sex', 'smoking_status', 'pleural_effusion', 'nodule_greater_4mm', 'emphysema', 'fibrosis'

    config = {
        'batch_size': 256,
        'learning_rate': 1e-4,
        'epochs': 30,
        'patience': 15,
        'input_channels': 1,
        'num_features': 792,
        'num_classes': 2,  # Binary classification for all tasks
        'weight_decay': 1e-5,
        'train_json': 'train_data.json',  # Path to training JSON
        'val_json': 'val_data.json',      # Path to validation JSON
        'extract_features': False,  # Set to True to force re-extraction even if files exist
        'checkpoint_name': f"{checkpoint_path.split('/')[-1]}", # Name for saving features
        'task_name': TASK_NAME
    }
    
    # Initialize wandb (optional)
    # wandb.init(project="sex-classification", config=config)
    
    # Load data from JSON files
    with open("/pool/data/lung/NLST/json/mae_train_tab_data.json", 'r') as f:
        train_data = json.load(f)
    with open('/pool/data/lung/NLST/json/mae_dev_tab_data.json', 'r') as f:
        val_data = json.load(f)
    with open('/pool/data/lung/NLST/json/mae_test_tab_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Check if we need to extract features or if they already exist
    checkpoint_name = config['checkpoint_name']
    task_name = config['task_name']
    
    train_cls, train_labels = load_cls_tokens(checkpoint_name, 'train', task_name)
    val_cls, val_labels = load_cls_tokens(checkpoint_name, 'val', task_name)
    test_cls, test_labels = load_cls_tokens(checkpoint_name, 'test', task_name)
    
    if config['extract_features'] or train_cls is None or val_cls is None or test_cls is None:
        print("Extracting CLS tokens from images...")
        
        # Create transforms for feature extraction
        extract_transforms = create_transforms(train=False)  # No augmentation for feature extraction
        
        # Create MONAI datasets for feature extraction
        train_dataset = Dataset(data=train_data, transform=extract_transforms)
        val_dataset = Dataset(data=val_data, transform=extract_transforms)
        test_dataset = Dataset(data=test_data, transform=extract_transforms)
        
        # Create dataloaders for feature extraction
        extract_batch_size = 32  # Smaller batch size for feature extraction to avoid OOM
        train_extract_loader = DataLoader(
            train_dataset, 
            batch_size=extract_batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        val_extract_loader = DataLoader(
            val_dataset, 
            batch_size=extract_batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        test_extract_loader = DataLoader(
            test_dataset, 
            batch_size=extract_batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model for feature extraction
        model = Lungevity(
            transformer='eva',
            patch_size=[8,8,8],
            grid_size=[
                int([160, 240, 128][0]/[8,8,8][0]), 
                int([160, 240, 128][1]/[8,8,8][1]), 
                int([160, 240, 128][2]/[8,8,8][2])
                ],
            enc_dim=792,
            enc_blocks=12,
            enc_heads=12,
            dropout_rate=0,
            num_reg_tokens=0,
            use_cls=True,
            hidden_dim=792,
            max_followup=6,
            fusion_layer=False,
            guided_attention_heads=4,
            use_mean_token=False,
                        ).to(device)
        
        # Load pre-trained checkpoint with strict=False
        checkpoint_path = checkpoint_path
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Extract state dict from checkpoint (handle different checkpoint formats)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # Load with strict=False to handle key mismatches
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"Missing keys in checkpoint: {len(missing_keys)} keys")
                    print(f"First few missing keys: {missing_keys[:5]}")
                
                if unexpected_keys:
                    print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                    print(f"First few unexpected keys: {unexpected_keys[:5]}")
                
                print("Successfully loaded checkpoint with strict=False")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with randomly initialized weights...")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            print("Continuing with randomly initialized weights...")
        
        # Extract CLS tokens for all splits
        extract_cls_tokens(model, train_extract_loader, 'train', checkpoint_name, device)
        extract_cls_tokens(model, val_extract_loader, 'val', checkpoint_name, device)
        extract_cls_tokens(model, test_extract_loader, 'test', checkpoint_name, device)
        
        # Load the extracted features
        train_cls, train_labels = load_cls_tokens(checkpoint_name, 'train', task_name)
        val_cls, val_labels = load_cls_tokens(checkpoint_name, 'val', task_name)
        test_cls, test_labels = load_cls_tokens(checkpoint_name, 'test', task_name)
        
        # Clear the full model from memory
        del model
        torch.cuda.empty_cache()
        
        print("Feature extraction completed!")
    else:
        print("Loading pre-computed CLS tokens...")
        print(f"Train: {train_cls.shape}, Val: {val_cls.shape}, Test: {test_cls.shape}")
    
    # Create datasets from pre-computed CLS tokens
    train_dataset = CLSDataset(train_cls, train_labels, task_name)
    val_dataset = CLSDataset(val_cls, val_labels, task_name)
    test_dataset = CLSDataset(test_cls, test_labels, task_name)
    
    # Create dataloaders for training on CLS tokens
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,  # Now we can shuffle since we're not extracting features
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Now create a simple classifier for the CLS tokens
    class CLSClassifier(nn.Module):
        def __init__(self, input_dim=792, num_classes=2):
            super().__init__()
            self.classifier = nn.Linear(input_dim, num_classes)
        
        def forward(self, x):
            return self.classifier(x)
    
    # Initialize the simple classifier
    model = CLSClassifier(input_dim=792, num_classes=config['num_classes']).to(device)
    
    # Print which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    criterion = nn.CrossEntropyLoss()  # For classification
    # Only pass trainable parameters to optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Starting training for {config['epochs']} epochs on task: {task_name}...")
    
    for epoch in range(config['epochs']):
        # Training phase
        train_losses = []
        train_preds = []
        train_targets = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for batch in train_pbar:
            loss, preds, targets = train_step(model, batch, criterion, optimizer, scaler, task_name)
            
            train_losses.append(loss)
            train_preds.append(preds)
            train_targets.append(targets)
            
            train_pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Validation phase
        val_losses = []
        val_preds = []
        val_targets = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
        for batch in val_pbar:
            loss, preds, targets = validate_step(model, batch, criterion, task_name)
            
            val_losses.append(loss)
            val_preds.append(preds)
            val_targets.append(targets)
            
            val_pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Compute epoch metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        train_metrics = compute_metrics(train_targets, train_preds)
        val_metrics = compute_metrics(val_targets, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1']:.3f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1']:.3f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to wandb (optional)
        # wandb.log({
        #     'epoch': epoch,
        #     'train/loss': train_loss,
        #     'train/accuracy': train_metrics['accuracy'],
        #     'train/f1': train_metrics['f1'],
        #     'val/loss': val_loss,
        #     'val/accuracy': val_metrics['accuracy'],
        #     'val/f1': val_metrics['f1'],
        #     'lr': optimizer.param_groups[0]['lr']
        # })
        
        # Early stopping and checkpointing (using accuracy instead of loss)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_metrics': val_metrics,
                'task_name': task_name
            }, f'best_{task_name}_model.pth')
            
            print(f"New best model saved! Val Accuracy: {val_metrics['accuracy']:.3f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 60)
    
    print("Training completed!")
    
    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(f'best_{task_name}_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    model.eval()
    final_preds = []
    final_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final evaluation"):
            cls_tokens = batch['cls'].to(device)
            labels = batch[task_name].to(device).long()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                logits = model(cls_tokens)
                probs = F.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
            
            final_preds.append(predictions.cpu().numpy())
            final_targets.append(labels.cpu().numpy())
    
    final_preds = np.concatenate(final_preds)
    final_targets = np.concatenate(final_targets)
    final_metrics = compute_metrics(final_targets, final_preds)
    
    print(f"\nFinal Results for {task_name.upper()}:")
    print(f"Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"Precision: {final_metrics['precision']:.3f}")
    print(f"Recall: {final_metrics['recall']:.3f}")
    print(f"F1-Score: {final_metrics['f1']:.3f}")
    
    # Print detailed classification report
    print(f"\nDetailed Classification Report for {task_name.upper()}:")
    
    # Define class names based on task
    class_names = {
        'sex': ['Female', 'Male'],
        'smoking_status': ['Non-Smoker', 'Smoker'], 
        'pleural_effusion': ['No Effusion', 'Effusion'],
        'nodule_greater_4mm': ['No Nodule >4mm', 'Nodule >4mm'],
        'emphysema': ['No Emphysema', 'Emphysema'],
        'fibrosis': ['No Fibrosis', 'Fibrosis']
    }
    
    target_names = class_names.get(task_name, ['Class 0', 'Class 1'])
    print(classification_report(final_targets, final_preds, target_names=target_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(final_targets, final_preds)
    print(cm)

if __name__ == "__main__":
    main()
