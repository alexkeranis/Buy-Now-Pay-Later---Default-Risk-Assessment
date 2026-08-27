# src/utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

def knn_impute_train_test(train_df, test_df, num_cols, cat_cols, n_neighbors=5):
    """
    Impute missing values in train and test sets using KNN (median).
    Fits StandardScaler, OneHotEncoder, and KNNImputer ONLY on train set.
    Returns imputed DataFrames (original scale for numerics).
    """
    X_train = train_df[num_cols + cat_cols].copy()
    X_test = test_df[num_cols + cat_cols].copy()

    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), num_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_cols)
    ])
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
    imputer.fit(X_train_processed)

    X_train_imputed_processed = imputer.transform(X_train_processed)
    X_test_imputed_processed = imputer.transform(X_test_processed)

    n_num_cols = len(num_cols)
    train_num_scaled = X_train_imputed_processed[:, :n_num_cols]
    test_num_scaled = X_test_imputed_processed[:, :n_num_cols]

    scaler = preprocessor.named_transformers_['scale']
    train_num_imputed = scaler.inverse_transform(train_num_scaled)
    test_num_imputed = scaler.inverse_transform(test_num_scaled)

    train_imputed = train_df.copy()
    test_imputed = test_df.copy()
    for i, col in enumerate(num_cols):
        train_imputed[col] = train_num_imputed[:, i]
        test_imputed[col] = test_num_imputed[:, i]

    return train_imputed, test_imputed

def load_raw_data(file_path):
    return pd.read_csv(file_path)

def quality_check(df, num_cols):
    """Print dataset overview, missing, duplicates, garbage values."""
    print("="*60)
    print("Dataset Overview")
    df.info()
    print("\n" + "="*60)
    print("Data Quality Check:")
    print("Missing values, duplicates & garbage values")
    print("="*60)
    total = len(df)
    complete = df.dropna().shape[0]
    print(f"Total samples: {total}")
    print(f"Complete rows: {complete}")
    print(f"Rows with missing: {total - complete}")
    print(f"Percentage complete: {complete/total*100:.2f}%")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nColumns with missing values:")
        print(missing)
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("Garbage values (negative/out‑of‑range):")
    for col in num_cols:
        if col == 'Age':
            invalid = (df[col] < 18) | (df[col] > 100)
        elif col == 'Credit_Score':
            invalid = (df[col] < 0) | (df[col] > 1000)
        else:
            invalid = (df[col] < 0)
        print(f"  {col}: {invalid.sum()} invalid")
    infs = df[num_cols].isin([np.inf, -np.inf]).sum().sum()
    if infs:
        print(f"  Warning: {infs} infinite values.")

def plot_histograms(df, num_cols, figsize=(15,10)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        kde_flag = False if col == 'Total_BNPL_Active_Loans' else True
        sns.histplot(data, kde=kde_flag, ax=axes[i], color='steelblue', bins=30)
        axes[i].set_title(f'{col} (raw)\n(non-null: {len(data)})')
        axes[i].set_xlabel('')
    plt.tight_layout()
    return fig, axes

def plot_boxplots(df, num_cols, target_col, figsize=(15,10)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        plot_df = df[[col, target_col]].dropna()
        sns.boxplot(x=target_col, y=col, data=plot_df, ax=axes[i],
                    hue=target_col, palette='Set2', legend=False)
        axes[i].set_title(f'{col} vs Default_Risk (raw)')
        axes[i].set_xlabel('')
    plt.tight_layout()
    return fig, axes

def categorical_crosstab(df, cat_cols, target_col, risk_order=['Low','Medium','High']):
    print("\nCategorical breakdown vs Default_Risk (as %):")
    for col in cat_cols:
        ctab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        ctab = ctab[risk_order]
        print(f"\n--- {col} ---")
        print(ctab.round(1))

def knn_impute_full(df, num_cols, cat_cols, n_neighbors=5):
    """Impute full dataset (for EDA) using KNN with median."""
    # Standardize numeric columns (using non-null stats)
    scaler_params = {}
    scaled_list = []
    for col in num_cols:
        mean = df[col].mean(skipna=True)
        std = df[col].std(skipna=True)
        scaler_params[col] = {'mean': mean, 'std': std}
        scaled = (df[col] - mean) / std
        scaled_list.append(scaled.values.reshape(-1,1))
    # One-hot encode categoricals
    cat_dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    X = np.concatenate(scaled_list, axis=1)
    X = np.concatenate([X, cat_dummies.values], axis=1)
    print(f"Feature matrix shape: {X.shape}")
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
    X_imp = imputer.fit_transform(X)
    missing_before = np.isnan(X).sum()
    missing_after = np.isnan(X_imp).sum()
    print(f"Missing before: {missing_before} | after: {missing_after}")
    # Extract imputed columns for Income and Credit
    income_idx = num_cols.index('Income_USD')
    credit_idx = num_cols.index('Credit_Score')
    imp_income_scaled = X_imp[:, income_idx]
    imp_credit_scaled = X_imp[:, credit_idx]
    imp_income = (imp_income_scaled * scaler_params['Income_USD']['std']) + scaler_params['Income_USD']['mean']
    imp_credit = (imp_credit_scaled * scaler_params['Credit_Score']['std']) + scaler_params['Credit_Score']['mean']
    df_imp = df.copy()
    df_imp['Income_USD'] = imp_income
    df_imp['Credit_Score'] = imp_credit
    return df_imp, scaler_params

def verify_imputation_stats(raw_df, imputed_df, cols=['Income_USD','Credit_Score']):
    print("\nVerifying Imputed Data Statistically")
    for col in cols:
        orig_mean = raw_df[col].mean(skipna=True)
        orig_median = raw_df[col].median(skipna=True)
        new_mean = imputed_df[col].mean()
        new_median = imputed_df[col].median()
        print(f"\n--- {col} ---")
        print(f"  Original (non-null) Mean: {orig_mean:.2f} | Median: {orig_median:.2f}")
        print(f"  Imputed (full) Mean:      {new_mean:.2f} | Median: {new_median:.2f}")
        print(f"  Difference in Mean:       {new_mean - orig_mean:.2f}")

def plot_imputation_overlay(raw_df, imputed_df, cols, figsize=(12,5)):
    fig, axes = plt.subplots(1, len(cols), figsize=figsize)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(raw_df[col].dropna(), kde=True, color='red', label='Raw (non-null)',
                     ax=ax, alpha=0.5, bins=30)
        sns.histplot(imputed_df[col], kde=True, color='blue', label='Imputed (full)',
                     ax=ax, alpha=0.5, bins=30)
        ax.set_title(f'{col}: Raw vs Imputed')
        ax.legend()
    plt.tight_layout()
    return fig, axes

def plot_correlation_heatmap(df, cols, figsize=(8,6)):
    plt.figure(figsize=figsize)
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink":0.8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()

def verify_robust_scaler(df, num_cols, id_col, target_col):
    """Scale numeric features with RobustScaler and print median/IQR."""
    X = df.drop(columns=[id_col, target_col])
    X_num = X[num_cols]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_num)
    scaled_df = pd.DataFrame(X_scaled, columns=num_cols)
    summary = scaled_df.describe().round(4)
    median = summary.loc['50%']
    iqr = summary.loc['75%'] - summary.loc['25%']
    verification_df = pd.DataFrame({'median': median, 'IQR': iqr}).T
    print("Verification (should have median ≈ 0, IQR ≈ 1):")
    with pd.option_context('display.show_dimensions', False):
        print(verification_df.round(4))
    return scaler, scaled_df

def prepare_and_save_data(df_raw, target_col, id_col, num_cols, cat_cols,
                          test_size=0.2, save_path='data/processed/final_data.npz',
                          save=True):
    """Stratified split, KNN imputation, add Debt_to_Income_Ratio, scale.
       If save=True (default), save to npz. Otherwise, just return processed arrays."""
    # Split
    y = df_raw[target_col]
    train_raw, test_raw = train_test_split(df_raw, test_size=test_size,
                                           random_state=42, stratify=y)
    print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

    # Leakage‑free KNN imputation
    train_imp, test_imp = knn_impute_train_test(train_raw, test_raw, num_cols, cat_cols)

    # Add Debt_to_Income_Ratio
    train_imp['Debt_to_Income_Ratio'] = train_imp['Total_BNPL_Debt_USD'] / train_imp['Income_USD'].replace(0, np.nan)
    test_imp['Debt_to_Income_Ratio'] = test_imp['Total_BNPL_Debt_USD'] / test_imp['Income_USD'].replace(0, np.nan)
    num_cols_with_ratio = num_cols + ['Debt_to_Income_Ratio']

    # RobustScaler (fit on train only)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_imp[num_cols_with_ratio])
    X_test_scaled = scaler.transform(test_imp[num_cols_with_ratio])

    # Encode categoricals for PyTorch (consistent mapping)
    cat_train_raw = train_raw[cat_cols].copy()
    cat_test_raw = test_raw[cat_cols].copy()
    for col in cat_cols:
        combined = pd.concat([cat_train_raw[col], cat_test_raw[col]], axis=0)
        encoder = pd.Categorical(combined)
        cat_train_raw[col] = encoder.codes[:len(cat_train_raw)]
        cat_test_raw[col] = encoder.codes[len(cat_train_raw):]
    cat_train = cat_train_raw.values
    cat_test = cat_test_raw.values
    cat_cardinalities = [train_raw[col].nunique() for col in cat_cols]

    # Save if requested
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path,
                 X_train_scaled=X_train_scaled,
                 X_test_scaled=X_test_scaled,
                 y_train=train_imp[target_col].values,
                 y_test=test_imp[target_col].values,
                 cat_train=cat_train,
                 cat_test=cat_test,
                 cat_cardinalities=np.array(cat_cardinalities),
                 cat_cols=np.array(cat_cols),
                 num_cols=np.array(num_cols),
                 target_col=target_col)
        print(f"Data saved to {save_path}")
    else:
        print("Data prepared but not saved (save=False).")

    # Return the prepared objects (so we can use them in notebooks)
    return (train_imp, test_imp, X_train_scaled, X_test_scaled,
            cat_train, cat_test, cat_cardinalities, num_cols_with_ratio)

# ------------------- Data loading -------------------
def load_processed_data(data_path='data/processed/final_data.npz'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cache not found: {data_path}. Run data_prep.py first.")
    data = np.load(data_path, allow_pickle=True)
    return {
        'X_train': data['X_train_scaled'],
        'X_test': data['X_test_scaled'],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'cat_train': data['cat_train'],
        'cat_test': data['cat_test'],
        'cat_cardinalities': data['cat_cardinalities'],
        'cat_cols': data['cat_cols'].tolist(),
        'num_cols': data['num_cols'].tolist(),
        'target_col': str(data['target_col'])
    }

def encode_target(y, mapping={'Low':0, 'Medium':1, 'High':2}):
    return np.array([mapping[x] for x in y])

# ------------------- PyTorch Dataset -------------------
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

# ------------------- Model definition -------------------
class EmbeddingNet(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, emb_dim=8, hidden_dims=[128,64]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
        total_emb_dim = len(cat_cardinalities) * emb_dim
        total_input_dim = num_numeric + total_emb_dim
        layers = []
        prev_dim = total_input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.4)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat):
        cat_embs = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        x = torch.cat([x_num, cat_embs], dim=1)
        return self.net(x)

# ------------------- Training loop -------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                device, epochs=200, patience=15):
    model.to(device)
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_num, X_cat, y in train_loader:
            X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_num, X_cat), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_num.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for X_num, X_cat, y in test_loader:
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                outputs = model(X_num, X_cat)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X_num.size(0)
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        val_f1 = f1_score(targets, preds, average='macro')
        val_f1s.append(val_f1)

        scheduler.step(val_loss)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {lr:.2e}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"Best validation Macro F1: {best_val_f1:.4f}")
    return model, train_losses, val_losses, val_f1s

# ------------------- Evaluation helpers -------------------
def evaluate_model(model, test_loader, device, target_names=['Low','Medium','High']):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_num, X_cat, y in test_loader:
            X_num, X_cat = X_num.to(device), X_cat.to(device)
            outputs = model(X_num, X_cat)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(y.numpy())
    report = classification_report(targets, preds, target_names=target_names, digits=4, output_dict=True)
    macro_f1 = f1_score(targets, preds, average='macro')
    return report, macro_f1, preds, targets

# ------------------- XGBoost training -------------------
def train_xgboost(X_train, y_train, X_test, y_test):
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3,
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_test, y_test)], verbose=False)
    return model