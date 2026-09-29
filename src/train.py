import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, DATA_PATH, DROPOUT, EMB_DIM, EPOCHS, HIDDEN_DIMS,
    LEARNING_RATE, MODEL_CONFIG_PATH, MODEL_PATH, PATIENCE, RANDOM_STATE,
    WEIGHT_DECAY,
)
from utils import (
    EmbeddingNet, TabularDataset, encode_target, evaluate_model,
    load_processed_data, train_model,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(RANDOM_STATE)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PATH}. "
            "Run src/data_preparation.py first."
        )

    data = load_processed_data(DATA_PATH)
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    cat_train, cat_val, cat_test = data["cat_train"], data["cat_val"], data["cat_test"]
    y_train = encode_target(data["y_train"])
    y_val = encode_target(data["y_val"])
    y_test = encode_target(data["y_test"])
    cat_cardinalities = data["cat_cardinalities"]

    train_loader = DataLoader(
        TabularDataset(X_train, cat_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TabularDataset(X_val, cat_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        TabularDataset(X_test, cat_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingNet(
        num_numeric=X_train.shape[1],
        cat_cardinalities=cat_cardinalities,
    )

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 3

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    model, train_losses, val_losses, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs=EPOCHS, patience=PATIENCE,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    model_config = {
        "num_numeric": int(X_train.shape[1]),
        "cat_cardinalities": [int(c) for c in cat_cardinalities],
        "emb_dim": EMB_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
    }
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(model_config, f, indent=2)

    report, macro_f1, _, _ = evaluate_model(model, test_loader, device)
    print(f"Test Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()