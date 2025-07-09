import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    mean_squared_error,
    r2_score
)

def task_evaluation(
    data_train, data_dev,
    label_train, label_dev,
    task, epochs, batch_size,
    problem_type="classification"  # New parameter
):
    """
    Trains a simple linear model on the provided training embeddings and evaluates it on the validation embeddings.
    
    Parameters:
    - data_train: List or iterable of training embeddings.
    - data_dev: List or iterable of validation embeddings.
    - label_train: List or iterable of training labels.
    - label_dev: List or iterable of validation labels.
    - task: String identifier for the task (used in logging).
    - epochs: Number of training epochs.
    - batch_size: Batch size for the DataLoader.
    - problem_type: "classification" or "regression". Default is "classification".

    Returns:
    - None. Logs metrics to Weights & Biases and prints final evaluation metrics.
    """

    # ---------------------------
    # 1. Data Preparation and Task specification
    # ---------------------------
    # Unbatch tensors first
    regression_tasks = [
                    'age', 
                    'smoking_duration'
                    ]
    classification_tasks = [
                            'smoking_status',
                            'gender'
                            'age',
                            'COPD',
                            'nodule_size_greater_than_15mm',
                            'nodule_size_greater_than_10mm',
                            'nodule_size_greater_than 5mm',
                            'visible_nodule_any_size'
                            ]

    if task in regression_tasks:
        problem_type = 'regression'
    else:
        problem_type = 'classification'
    
    cls_embeddings_train = torch.cat(data_train, dim=0).float()  # Shape: [total_samples, embed_dim]
    cls_embeddings_val = torch.cat(data_dev, dim=0).float()
    
    # Convert labels to tensors
    if problem_type == "classification":
        out_dim = 2
        y_train = torch.tensor(label_train).long()
        y_dev = torch.tensor(label_dev).long()
    else:
        out_dim = 1
        y_train = torch.tensor(label_train).float().unsqueeze(1)
        y_dev = torch.tensor(label_dev).float().unsqueeze(1)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_embeddings_train = cls_embeddings_train.to(device)
    cls_embeddings_val = cls_embeddings_val.to(device)
    y_train = y_train.to(device)
    y_dev = y_dev.to(device)

    # ---------------------------
    # 2. Model Definition
    # ---------------------------

    class SimpleModel(nn.Module):
        def __init__(self, input_dim, out_dim, seed=42):
            super().__init__()
            torch.manual_seed(seed)
            self.linear = nn.Linear(input_dim, out_dim)
            torch.nn.init.xavier_uniform_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias)
            
        def forward(self, x):
            return self.linear(x)

    model = SimpleModel(cls_embeddings_train.shape[1], out_dim).to(device)

    # ---------------------------
    # 3. Training Setup
    # ---------------------------

    # Define loss function according to the problem type
    if problem_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    # Create DataLoader
    train_loader = DataLoader(
        TensorDataset(cls_embeddings_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    # ---------------------------
    # 4. Training Loop
    # ---------------------------
    for epoch_num in tqdm(range(epochs), desc='Training embeddings...'):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

    # ---------------------------
    # 5. Evaluation
    # ---------------------------
    model.eval()
    with torch.no_grad():
        all_outputs = model(cls_embeddings_val)

    if problem_type == "classification":
        # Compute predictions
        _, predicted = torch.max(all_outputs, 1)
        y_true = y_dev.cpu()
        y_pred = predicted.cpu()
        
        # Check for number of unique classes
        unique_classes = len(torch.unique(y_true))
        
        if unique_classes > 1:
            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary'
            )
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Log and print
            wandb.log({
                f'dev/acc_{task}': accuracy,
                f'dev/specificity_{task}': specificity,
                f'dev/sensitivity_{task}': recall,
                f'dev/precision_{task}': precision,
                f'dev/f1_score_{task}': f1,
                f'dev/mcc_{task}': mcc,
            })
            print(f'MCC: {mcc}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        else:
            print(f'Skipping metrics for {task}: Only one class present in validation set')
        
        print('Done evaluating embeddings (Classification).')
        print('Continuing training...')
        
    else:  # Regression
        # For regression, the raw output is the prediction
        # Make sure to squeeze if you used shape [N,1]
        predicted = all_outputs.squeeze()
        y_true = y_dev.squeeze()
        
        # Compute metrics
        mse = mean_squared_error(y_true.cpu(), predicted.cpu())
        rmse = mse ** 0.5
        r2 = r2_score(y_true.cpu(), predicted.cpu())

        # Log and print
        wandb.log({
            f'dev/mse_{task}': mse,
            f'dev/rmse_{task}': rmse,
            f'dev/r2_{task}': r2
        })
        print('Done evaluating embeddings (Regression).')
        print(f'MSE: {mse}, RMSE: {rmse}, R2: {r2}')
        print('Continuing training...')
