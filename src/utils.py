import json
import pickle
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from torch.utils.data import Dataset

from config import (
    CAT_COLS, DROPOUT, EMB_DIM, FEATURE_COLS, HIDDEN_DIMS, N_NEIGHBORS,
    NUM_COLS, RANDOM_STATE, RATIO_COL, TARGET_COL, TARGET_MAPPING,
    TARGET_NAMES, TEST_SIZE, VAL_SIZE,
)


def load_raw_data(path):
    return pd.read_csv(path)


def _impute_numeric(train_df, other_dfs, num_cols, cat_cols, n_neighbors):
    """Fit ColumnTransformer + KNNImputer on train only; apply to train and others.

    Numeric columns are scaled, imputed, then inverse-transformed back to
    original units. Categoricals are left untouched; NaN handling for them
    happens later.
    """
    X_train = train_df[num_cols + cat_cols].copy()

    preprocessor = ColumnTransformer([
        ("scale", StandardScaler(), num_cols),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False, drop="first"), cat_cols),
    ])
    preprocessor.fit(X_train)

    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance",
                         metric="nan_euclidean")
    imputer.fit(preprocessor.transform(X_train))

    scaler = preprocessor.named_transformers_["scale"]
    n_num = len(num_cols)

    def apply(df):
        X = df[num_cols + cat_cols].copy()
        X_imp = imputer.transform(preprocessor.transform(X))
        nums = scaler.inverse_transform(X_imp[:, :n_num])
        out = df.copy()
        for i, col in enumerate(num_cols):
            out[col] = nums[:, i]
        return out

    return [apply(train_df)] + [apply(d) for d in other_dfs]


def _encode_categoricals(df, mappings):
    arr = np.zeros((len(df), len(CAT_COLS)), dtype=np.int64)
    for i, col in enumerate(CAT_COLS):
        arr[:, i] = df[col].map(mappings[col]).fillna(0).astype(np.int64).values
    return arr


def prepare_and_save_data(df_raw, save_path, save=True):
    """Stratified train/val/test split, KNN imputation, ratio feature,
    unified categorical encoding, robust scaling. Returns (result, scaler, mappings)."""
    y = df_raw[TARGET_COL]
    train_val_raw, test_raw = train_test_split(
        df_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    train_raw, val_raw = train_test_split(
        train_val_raw, test_size=val_ratio, random_state=RANDOM_STATE,
        stratify=train_val_raw[TARGET_COL],
    )

    train_imp, val_imp, test_imp = _impute_numeric(
        train_raw, [val_raw, test_raw], NUM_COLS, CAT_COLS, N_NEIGHBORS,
    )

    for df in (train_imp, val_imp, test_imp):
        df[RATIO_COL] = np.where(
            df["Income_USD"] > 0,
            df["Total_BNPL_Debt_USD"] / df["Income_USD"],
            0.0
        )
        for col in CAT_COLS:
            df[col] = df[col].fillna("Unknown")

    for name, df in (("train", train_imp), ("val", val_imp), ("test", test_imp)):
        assert not df[FEATURE_COLS].isnull().any().any(), f"NaNs remain in {name} features"

    cat_mappings = {}
    for col in CAT_COLS:
        categories = pd.Categorical(train_imp[col]).categories
        cat_mappings[col] = {cat: i + 1 for i, cat in enumerate(categories)}

    cat_train = _encode_categoricals(train_imp, cat_mappings)
    cat_val = _encode_categoricals(val_imp, cat_mappings)
    cat_test = _encode_categoricals(test_imp, cat_mappings)

    cat_cardinalities = [len(cat_mappings[col]) + 1 for col in CAT_COLS]

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_imp[FEATURE_COLS])
    X_val = scaler.transform(val_imp[FEATURE_COLS])
    X_test = scaler.transform(test_imp[FEATURE_COLS])

    result = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": train_imp[TARGET_COL].values,
        "y_val": val_imp[TARGET_COL].values,
        "y_test": test_imp[TARGET_COL].values,
        "cat_train": cat_train, "cat_val": cat_val, "cat_test": cat_test,
        "cat_cardinalities": np.array(cat_cardinalities),
        "feature_cols": np.array(FEATURE_COLS),
        "cat_cols": np.array(CAT_COLS),
        "target_col": TARGET_COL,
    }

    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **result)

    return result, scaler, cat_mappings


def load_processed_data(path):
    data = np.load(path, allow_pickle=True)
    return {
        "X_train": data["X_train"], "X_val": data["X_val"], "X_test": data["X_test"],
        "y_train": data["y_train"], "y_val": data["y_val"], "y_test": data["y_test"],
        "cat_train": data["cat_train"], "cat_val": data["cat_val"], "cat_test": data["cat_test"],
        "cat_cardinalities": data["cat_cardinalities"],
        "feature_cols": data["feature_cols"].tolist(),
        "cat_cols": data["cat_cols"].tolist(),
        "target_col": str(data["target_col"]),
    }


def encode_target(y):
    return np.array([TARGET_MAPPING[x] for x in y], dtype=np.int64)


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class EmbeddingNet(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, emb_dim=EMB_DIM,
                 hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(c), emb_dim) for c in cat_cardinalities]
        )
        total_input_dim = num_numeric + len(cat_cardinalities) * emb_dim
        layers = []
        prev = total_input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        cat_embs = torch.cat(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )
        return self.net(torch.cat([x_num, cat_embs], dim=1))


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, patience):
    model.to(device)
    best_f1 = 0.0
    best_state = None
    patience_counter = 0
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for X_num, X_cat, y in train_loader:
            X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_num, X_cat), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X_num.size(0)
        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for X_num, X_cat, y in val_loader:
                X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
                out = model(X_num, X_cat)
                running += criterion(out, y).item() * X_num.size(0)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                targets.extend(y.cpu().numpy())
        val_loss = running / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_f1 = f1_score(targets, preds, average="macro")
        val_f1s.append(val_f1)

        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                  f"Val F1 {val_f1:.4f} | LR {lr:.2e}")

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_f1s


def evaluate_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_num, X_cat, y in loader:
            X_num, X_cat = X_num.to(device), X_cat.to(device)
            out = model(X_num, X_cat)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(y.numpy())
    report = classification_report(targets, preds, target_names=TARGET_NAMES,
                                   digits=4, output_dict=True)
    macro_f1 = f1_score(targets, preds, average="macro")
    return report, macro_f1, preds, targets


def load_inference_components(model_path, scaler_path, cat_encoder_path,
                              model_config_path):
    with open(model_config_path) as f:
        cfg = json.load(f)

    model = EmbeddingNet(
        num_numeric=cfg["num_numeric"],
        cat_cardinalities=cfg["cat_cardinalities"],
        emb_dim=cfg["emb_dim"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(cat_encoder_path, "rb") as f:
        cat_mappings = pickle.load(f)

    return model, scaler, cat_mappings


def preprocess_single_sample(input_dict, feature_cols, cat_cols, scaler, cat_mappings):
    X_num_df = pd.DataFrame(
        [[float(input_dict[c]) for c in feature_cols]],
        columns=feature_cols,
    )
    X_num_scaled = scaler.transform(X_num_df)

    X_cat = np.zeros((1, len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        val = input_dict[col]
        mapping = cat_mappings[col]
        if val not in mapping:
            warnings.warn(f"Unknown category '{val}' for {col}; using code 0")
        X_cat[0, i] = mapping.get(val, 0)
    return X_num_scaled, X_cat


def predict_single(model, X_num, X_cat, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_num_t = torch.tensor(X_num, dtype=torch.float32).to(device)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long).to(device)
        out = model(X_num_t, X_cat_t)
        probs = torch.softmax(out, dim=1).cpu().numpy().flatten()
        pred = int(np.argmax(probs))
    return pred, probs