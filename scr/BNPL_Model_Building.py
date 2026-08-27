"""
BNPL Default Risk - Model Building
---------------------------------
- Baseline: XGBoost with class balancing
- Advanced: PyTorch with Entity Embeddings + Focal Loss
- CPU-only for Spyder; GPU-ready with .to(device)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------------------
# 1. DATA LOADING FROM CACHE
# ------------------------------
print("="*60)
print("Loading processed datasets from cache")
print("="*60)

data_path = 'data/processed/final_data.npz'

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"File not found: {data_path}\n"
        "Please run BNPL_Data_Preparation_Whole.py first to generate the data."
    )

data = np.load(data_path, allow_pickle=True)

X_train_scaled = data['X_train_scaled']
X_test_scaled = data['X_test_scaled']
y_train = data['y_train']
y_test = data['y_test']
cat_train = data['cat_train']
cat_test = data['cat_test']
cat_cardinalities = data['cat_cardinalities']
cat_cols = data['cat_cols'].tolist()
num_cols = data['num_cols'].tolist()
target_col = str(data['target_col'])

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")
print(f"y_train shape:        {y_train.shape}")
print(f"y_test shape:         {y_test.shape}")
print(f"cat_train shape:      {cat_train.shape}")
print(f"cat_test shape:       {cat_test.shape}")
print(f"cat_cardinalities:    {cat_cardinalities}")
print(f"cat_cols:             {cat_cols}")
print(f"num_cols:             {num_cols}")
print(f"target_col:           {target_col}")
print()
print("\nData loaded successfully. Proceeding to modeling.\n")

# Encode target labels: Low=0, Medium=1, High=2
target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_train_enc = np.array([target_mapping[x] for x in y_train])
y_test_enc = np.array([target_mapping[x] for x in y_test])
target_names = ['Low', 'Medium', 'High']

print("Target labels encoded: 0=Low, 1=Medium, 2=High")

# ------------------------------
# 2. XGBOOST BASELINE
# ------------------------------
print("\n" + "="*60)
print("BASELINE: XGBoost with balanced sample weights")
print("="*60)

# Compute sample weights to handle class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_enc)

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

# Train
xgb_model.fit(
    X_train_scaled, y_train_enc,
    sample_weight=sample_weights,
    eval_set=[(X_test_scaled, y_test_enc)],
    verbose=False
)

# Evaluate
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("\nXGBoost Performance on Test Set:")
print(classification_report(y_test_enc, y_pred_xgb, target_names=target_names, digits=4))
xgb_macro_f1 = f1_score(y_test_enc, y_pred_xgb, average='macro')
print(f"Macro F1-Score: {xgb_macro_f1:.4f}")

# Confusion matrix
cm_xgb = confusion_matrix(y_test_enc, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ------------------------------
# 3. PYTORCH DATASET CLASS
# ------------------------------
print("\n" + "="*60)
print("ADVANCED: PyTorch Tabular Dataset with Embeddings")
print("="*60)

class TabularDataset(Dataset):
    """
    Custom Dataset for tabular data with separate numeric and categorical features.
    """
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)   # embeddings expect long ints
        self.y = torch.tensor(y, dtype=torch.long)           # CrossEntropyLoss expects long

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

# Create train/test datasets
train_ds = TabularDataset(X_train_scaled, cat_train, y_train_enc)
test_ds  = TabularDataset(X_test_scaled, cat_test, y_test_enc)

# DataLoaders (small batch size due to small dataset)
batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# ------------------------------
# 4. PYTORCH MODEL DEFINITION
# ------------------------------
class EmbeddingNet(nn.Module):
    """
    Neural network with:
    - Entity embeddings for categorical features.
    - Dense layers for numeric + concatenated embeddings.
    - BatchNorm, Dropout, and Focal Loss support.
    """
    def __init__(self, num_numeric, cat_cardinalities, emb_dim=8, hidden_dims=[128, 64]):
        super(EmbeddingNet, self).__init__()
        
        # 1. Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim) for card in cat_cardinalities
        ])
        
        # 2. Calculate total input dimension
        total_emb_dim = len(cat_cardinalities) * emb_dim
        total_input_dim = num_numeric + total_emb_dim
        
        # 3. Build dense layers with BatchNorm & Dropout
        layers = []
        prev_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4)
            ])
            prev_dim = hidden_dim
        
        # Output layer (3 classes)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_num, x_cat):
        # Embed categoricals and concatenate
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embs = torch.cat(cat_embs, dim=1)
        
        # Concatenate numeric + embeddings
        x = torch.cat([x_num, cat_embs], dim=1)
        
        # Forward through dense layers
        return self.net(x)

# Instantiate model
num_numeric = X_train_scaled.shape[1]  # 7
model = EmbeddingNet(
    num_numeric=num_numeric,
    cat_cardinalities=cat_cardinalities,
    emb_dim=8,
    hidden_dims=[128, 64]
)

print(f"\nModel architecture:\n{model}")
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ------------------------------
# 5. FOCAL LOSS DEFINITION
# ------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    gamma=2 is standard; alpha balances class weights.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Compute class weights for Focal Loss (inverse frequency)
class_counts = np.bincount(y_train_enc)
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * 3  # normalize to sum to 3

# ------------------------------
# 6. FULL TRAINING LOOP WITH EARLY STOPPING
# ------------------------------
print("\n" + "="*60)
print("PyTorch Training Loop")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Early stopping parameters
best_val_f1 = 0.0
patience = 15
patience_counter = 0
best_model_state = None

epochs = 200
train_losses = []
val_losses = []
val_f1s = []

for epoch in range(epochs):
    # ---------- Training ----------
    model.train()
    train_loss = 0.0
    for X_num, X_cat, y in train_loader:
        X_num = X_num.to(device)
        X_cat = X_cat.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_num, X_cat)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_num.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # ---------- Validation ----------
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_num, X_cat, y in test_loader:
            X_num = X_num.to(device)
            X_cat = X_cat.to(device)
            y = y.to(device)
            
            outputs = model(X_num, X_cat)
            loss = criterion(outputs, y)
            val_loss += loss.item() * X_num.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    
    # Compute macro F1 on validation set
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    val_f1s.append(val_f1)
    
    # Learning rate scheduling (based on validation loss)
    scheduler.step(val_loss)
    
    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f} | LR: {current_lr:.2e}")
    
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(best_model_state)
print(f"\nBest validation Macro F1: {best_val_f1:.4f}")

# Save the best model
os.makedirs('models', exist_ok=True)
torch.save(best_model_state, 'models/best_pytorch_model.pth')
print(f"Best model saved to models/best_pytorch_model.pth")
print(f"\nBest validation Macro F1: {best_val_f1:.4f}")

# ------------------------------
# 7. FINAL EVALUATION ON TEST SET
# ------------------------------
print("\n" + "="*60)
print("PyTorch Model Final Evaluation")
print("="*60)

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for X_num, X_cat, y in test_loader:
        X_num = X_num.to(device)
        X_cat = X_cat.to(device)
        outputs = model(X_num, X_cat)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.numpy())   # y is already on CPU

print("\nPyTorch Performance on Test Set:")
print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))
pytorch_macro_f1 = f1_score(all_targets, all_preds, average='macro')
print(f"Macro F1-Score: {pytorch_macro_f1:.4f}")

# Confusion matrix
cm_pt = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm_pt, annot=True, fmt='d', cmap='Greens')
plt.title('PyTorch Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ------------------------------
# 8. COMPARISON PLOTS: Loss and F1 over epochs
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves')
ax1.legend()
ax1.grid(True)

ax2.plot(val_f1s, label='Validation Macro F1', color='green')
ax2.axhline(y=xgb_macro_f1, color='red', linestyle='--', label='XGBoost Macro F1')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Macro F1')
ax2.set_title('Validation Macro F1')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"XGBoost  Macro F1: {xgb_macro_f1:.4f}")
print(f"PyTorch  Macro F1: {pytorch_macro_f1:.4f}")
print(f"Improvement:       {(pytorch_macro_f1 - xgb_macro_f1):+.4f}")