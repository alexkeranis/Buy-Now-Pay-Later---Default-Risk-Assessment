# src/train.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(__file__))
from utils import *

# ============================================================
# 0. CONFIGURATION AND WORKING DIRECTORY
# ============================================================
print("="*60)
print("TRAINING SCRIPT – BNPL Default Risk")
print("="*60)
print(f"Working directory: {os.getcwd()}")

# If running from inside src/, go up one level to project root
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
    print("Changed working directory to project root:", os.getcwd())

# Verify cache exists
cache_path = 'data/processed/final_data.npz'
if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f"Cache not found at {cache_path}.\n"
        "Please run src/data_preparation.py first."
    )
print("Cache found. Proceeding.\n")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("="*60)
print("STEP 1: Loading processed data from cache")
print("="*60)
data = load_processed_data()   # uses default path, which now resolves correctly
X_train, X_test = data['X_train'], data['X_test']
cat_train, cat_test = data['cat_train'], data['cat_test']
y_train = data['y_train']          # original string labels
y_test = data['y_test']
y_train_enc = encode_target(y_train)
y_test_enc = encode_target(y_test)
cat_cardinalities = data['cat_cardinalities']
target_names = ['Low', 'Medium', 'High']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"cat_train shape: {cat_train.shape}")
print(f"cat_test shape:  {cat_test.shape}")
print(f"y_train_enc shape: {y_train_enc.shape}")
print(f"y_test_enc shape:  {y_test_enc.shape}")
print("Sample y_train_enc[:10]:", y_train_enc[:10])
print()

# ============================================================
# 2. XGBOOST BASELINE
# ============================================================
print("="*60)
print("STEP 2: XGBoost Baseline")
print("="*60)
xgb_model = train_xgboost(X_train, y_train_enc, X_test, y_test_enc)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Performance on Test Set:")
print(classification_report(y_test_enc, y_pred_xgb, target_names=target_names, digits=4))
xgb_report = classification_report(y_test_enc, y_pred_xgb,
                                   target_names=target_names,
                                   digits=4, output_dict=True)
xgb_f1 = xgb_report['macro avg']['f1-score']
print(f"XGBoost Macro F1: {xgb_f1:.4f}\n")

# ============================================================
# 3. PYTORCH SETUP
# ============================================================
print("="*60)
print("STEP 3: PyTorch Setup (DataLoaders, Model, Optimizer)")
print("="*60)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

batch_size = 256
train_ds = TabularDataset(X_train, cat_train, y_train_enc)
test_ds  = TabularDataset(X_test,  cat_test,  y_test_enc)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = EmbeddingNet(num_numeric=X_train.shape[1],
                     cat_cardinalities=cat_cardinalities)
print(f"Model architecture:\n{model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Class weights for CrossEntropy
class_counts = np.bincount(y_train_enc)
class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * 3
print(f"Class weights (normalised): {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
print("Optimizer, criterion, scheduler initialised.\n")

# ============================================================
# 4. TRAIN PYTORCH MODEL
# ============================================================
print("="*60)
print("STEP 4: PyTorch Training Loop (with early stopping)")
print("="*60)
model, train_losses, val_losses, val_f1s = train_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler,
    device, epochs=200, patience=15
)

# ============================================================
# 5. SAVE BEST MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 5: Saving best model")
print("="*60)
os.makedirs('models', exist_ok=True)
model_path = 'models/best_pytorch_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path) / 1024:.1f} KB")
else:
    print("ERROR: Model file not saved.")
print()

# ============================================================
# 6. FINAL EVALUATION ON TEST SET
# ============================================================
print("="*60)
print("STEP 6: PyTorch Final Evaluation on Test Set")
print("="*60)
pytorch_report, pytorch_f1, preds, targets = evaluate_model(
    model, test_loader, device, target_names
)
print("\nPyTorch Performance on Test Set:")
print(classification_report(targets, preds, target_names=target_names, digits=4))
print(f"PyTorch Macro F1: {pytorch_f1:.4f}\n")

# ============================================================
# 7. FINAL COMPARISON
# ============================================================
print("="*60)
print("STEP 7: Final Comparison")
print("="*60)
print(f"XGBoost  Macro F1: {xgb_f1:.4f}")
print(f"PyTorch  Macro F1: {pytorch_f1:.4f}")
print(f"Improvement:       {pytorch_f1 - xgb_f1:+.4f}")
print("="*60)
print("✅ Training script completed.")