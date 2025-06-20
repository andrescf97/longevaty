import math
from functools import partial
from typing import Optional, Type, Final
import torch
from torch import nn
import torch.nn.functional as F
from tvital.sybil import SybilNet
from tvital.ctclip import CTViT

from timm.models.vision_transformer import LayerScale
from timm.layers import use_fused_attn
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp

class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()

        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
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
    

class Longevity(nn.Module):
    def __init__(
            self,
            encoder_model: str = "sybil",
            enc_hidden_dim: int = 512,
            hidden_dim: int = 512,
            max_followup: int = 6,
            blocks: int = 5,
            heads: int = 12,
            dropout_rate: float = 0.2,
            mlp_ratio: int = 4,
            longitundinal_model: str = "transformer",
    ):
        super(Longevity, self).__init__()
        if encoder_model == "sybil":
            self.encoder = SybilNet.load("/pool/users/chev/Sybil/checkpoints/65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt")
            self.encoder_head = nn.Identity()

        elif encoder_model == "ctrate":
            self.encoder  = CTViT(
                dim=512,
                codebook_size=8192,
                image_size=480,
                patch_size=20,
                temporal_patch_size=10,
                spatial_depth=4,
                temporal_depth=4,
                dim_head=32,
                heads=8
            )
            s = torch.load("/pool/data/lung/CT-RATE/models/CT-CLIP-Related/CT-CLIP_v2.pt", map_location="cpu", weights_only=False)
            encoder_state_dict = {k.replace("visual_transformer.", ""): v for k, v in s.items() if k.startswith("visual_transformer.")}
            self.encoder.load_state_dict(encoder_state_dict, strict=True)


        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.transformer = nn.ModuleList([
            Block(dim=enc_hidden_dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=True, 
                    attn_drop=dropout_rate, act_layer=nn.GELU)
            for _ in range(blocks)
        ])
        self.is_transformer = True
        self.final_norm = nn.LayerNorm(enc_hidden_dim, eps=1e-6)
        self.time_embedding = nn.Sequential(*[
            nn.Linear(enc_hidden_dim, enc_hidden_dim),
            nn.SELU(),
            nn.Linear(enc_hidden_dim, enc_hidden_dim)
        ])

        self.classifier = nn.Sequential(*[
            nn.Linear(enc_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            Cumulative_Probability_Layer(hidden_dim, max_followup)
        ])

    def forward(self, img0, img1, img2, t_mask, time_embed):
        emb0 = self.encoder_head(self.encoder(img0))
        emb1 = self.encoder_head(self.encoder(img1))
        emb2 = self.encoder_head(self.encoder(img2))

        batch = torch.stack((emb0, emb1, emb2), dim=1)
        B, _, _ = batch.shape

        batch = self.time_embedding(time_embed) + batch
        cls_token = self.cls.expand(B, -1, -1)
        x = torch.cat((cls_token, batch), dim=1)
        x = self.dropout(x)
        t_mask = torch.cat([torch.ones((x.shape[0], 1), dtype=torch.bool, device=t_mask.device), t_mask], dim=1)
        t_mask = t_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S+1]
        for layer in self.transformer:
            x = layer(x, mask=t_mask)
        x = self.final_norm(x)
        final_state = x[:, 0, :]
        op = self.classifier(final_state)
        return op


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def build_rel_time_embeddings(
    time: torch.Tensor,
    dim: int = 768,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Generates sinusoidal positional embeddings for scalar time values.
    Assumes `time` is always of shape [batch_size, sequence_length].

    Args:
        time (torch.Tensor): A tensor of scalar time values, shape [batch_size, sequence_length].
        dim (int): The dimension of the output sinusoidal embedding. Must be even.
        max_period (int): The maximum period for the sinusoidal functions.
        dtype (torch.dtype): The desired data type of the output embedding.

    Returns:
        torch.Tensor: The sinusoidal embeddings, shape [batch_size, sequence_length, dim].
    """
    if dim % 2 != 0:
        raise ValueError("Sinusoidal embedding dimension must be even.")

    original_shape = time.shape # This will now always be (B, S)
    
    time_flat = time.reshape(-1, 1).float() 

    half_dim = dim // 2
    epsilon = 1e-6
    exponent = math.log(max_period) / (half_dim - epsilon) if half_dim > 0 else 0.0
    
    frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=time_flat.device) * -exponent)
    args = time_flat * frequencies.unsqueeze(0) # unsqueeze(0) adds a dimension for broadcasting
    embeddings = torch.cat((torch.sin(args), torch.cos(args)), dim=-1) 
    return embeddings.reshape(*original_shape, dim)