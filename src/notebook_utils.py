import os
import utils as U
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from config import NUM_COLS, RANGES, TARGET_NAMES
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

def quality_check(df, num_cols=NUM_COLS):
    print("=" * 60)
    print("Dataset Overview")
    df.info()
    print("Data Quality Check:")
    print("Missing values, duplicates & garbage values")
    total = len(df)
    complete = df.dropna().shape[0]
    print(f"Total samples: {total}")
    print(f"Complete rows: {complete}")
    print(f"Rows with missing: {total - complete}")
    print(f"Percentage complete: {complete / total * 100:.2f}%")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nColumns with missing values:")
        print(missing)
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("Garbage values (negative/out-of-range):")
    for col in num_cols:
        low, high = RANGES[col]
        invalid = (df[col] < low) | (df[col] > high)
        print(f"  {col}: {invalid.sum()} invalid")
    infs = df[num_cols].isin([np.inf, -np.inf]).sum().sum()
    if infs:
        print(f"  Warning: {infs} infinite values.")
        
def plot_histograms(df, num_cols, figsize=(15, 10)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        kde_flag = False if col == "Total_BNPL_Active_Loans" else True
        sns.histplot(data, kde=kde_flag, ax=axes[i], color="steelblue", bins=30)
        axes[i].set_title(f"{col} (raw)\n(non-null: {len(data)})")
        axes[i].set_xlabel("")
    plt.tight_layout()
    return fig, axes

def plot_boxplots(df, num_cols, target_col, figsize=(15, 10)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        plot_df = df[[col, target_col]].dropna()
        sns.boxplot(x=target_col, y=col, data=plot_df, ax=axes[i],
                    hue=target_col, palette="Set2", legend=False)
        axes[i].set_title(f"{col} vs {target_col} (raw)")
        axes[i].set_xlabel("")
    plt.tight_layout()
    return fig, axes

def categorical_crosstab(df, cat_cols, target_col, risk_order=("Low", "Medium", "High")):
    print(f"\nCategorical breakdown vs {target_col} (as %):")
    for col in cat_cols:
        ctab = pd.crosstab(df[col], df[target_col], normalize="index") * 100
        ctab = ctab[list(risk_order)]
        print(f"\n--- {col} ---")
        print(ctab.round(1))
        
def knn_impute_full(df, num_cols, cat_cols, n_neighbors=5):
    """Full-dataset imputation for EDA only. Not leakage-free; do not use for modeling."""
    scaler_params = {}
    scaled_list = []
    for col in num_cols:
        mean = df[col].mean(skipna=True)
        std = df[col].std(skipna=True)
        scaler_params[col] = {"mean": mean, "std": std}
        scaled = (df[col] - mean) / std
        scaled_list.append(scaled.values.reshape(-1, 1))
    cat_dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    X = np.concatenate(scaled_list, axis=1)
    X = np.concatenate([X, cat_dummies.values], axis=1)
    print(f"Feature matrix shape: {X.shape}")
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance",
                         metric="nan_euclidean")
    X_imp = imputer.fit_transform(X)
    print(f"Missing before: {np.isnan(X).sum()} | after: {np.isnan(X_imp).sum()}")
    income_idx = num_cols.index("Income_USD")
    credit_idx = num_cols.index("Credit_Score")
    imp_income = X_imp[:, income_idx] * scaler_params["Income_USD"]["std"] + scaler_params["Income_USD"]["mean"]
    imp_credit = X_imp[:, credit_idx] * scaler_params["Credit_Score"]["std"] + scaler_params["Credit_Score"]["mean"]
    df_imp = df.copy()
    df_imp["Income_USD"] = imp_income
    df_imp["Credit_Score"] = imp_credit
    return df_imp, scaler_params

def verify_imputation_stats(raw_df, imputed_df, cols=("Income_USD", "Credit_Score")):
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
        
def plot_imputation_overlay(raw_df, imputed_df, cols, figsize=(12, 5)):
    fig, axes = plt.subplots(1, len(cols), figsize=figsize)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(raw_df[col].dropna(), kde=True, color="red",
                     label="Raw (non-null)", ax=ax, alpha=0.5, bins=30)
        sns.histplot(imputed_df[col], kde=True, color="blue",
                     label="Imputed (full)", ax=ax, alpha=0.5, bins=30)
        ax.set_title(f"{col}: Raw vs Imputed")
        ax.legend()
    plt.tight_layout()
    return fig, axes

def plot_correlation_heatmap(df, cols, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    
def verify_robust_scaler(df, feature_cols):
    """Fit RobustScaler on the full EDA frame; print median and IQR after scaling."""
    X_num = df[feature_cols]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_num)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    summary = scaled_df.describe().round(4)
    median = summary.loc["50%"]
    iqr = summary.loc["75%"] - summary.loc["25%"]
    verification_df = pd.DataFrame({"median": median, "IQR": iqr}).T
    print("Verification (should have median ≈ 0, IQR ≈ 1):")
    with pd.option_context("display.show_dimensions", False):
        print(verification_df.round(4))
    return scaler, scaled_df

def verify_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found at {path}. Run src/data_preparation.py first."
        )
    print("Cache found. Proceeding.")

def run_xgboost_baseline(X_train, y_train_enc, X_test, y_test_enc,
                         target_names=TARGET_NAMES):
    model = train_xgboost(X_train, y_train_enc, X_test, y_test_enc)
    y_pred = model.predict(X_test)

    print("XGBoost Performance on Test Set:")
    print(classification_report(y_test_enc, y_pred,
                                target_names=target_names, digits=4))
    macro_f1 = f1_score(y_test_enc, y_pred, average="macro")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(y_test_enc, y_pred,
                          title="XGBoost Counts",
                          normalize=None, cmap="Blues", ax=axs[0])
    plot_confusion_matrix(y_test_enc, y_pred,
                          title="XGBoost Row-Normalised",
                          normalize="true", cmap="Blues", ax=axs[1])
    plt.tight_layout()

    return model, y_pred, macro_f1

def run_pytorch_evaluation(model, test_loader, device,
                           target_names=TARGET_NAMES):
    report, macro_f1, preds, targets = U.evaluate_model(
        model, test_loader, device
    )

    print("PyTorch Performance on Test Set:")
    print(classification_report(targets, preds,
                                target_names=target_names, digits=4))
    print(f"Macro F1-Score: {macro_f1:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(targets, preds,
                          title="PyTorch Counts",
                          normalize=None, cmap="Greens", ax=axs[0])
    plot_confusion_matrix(targets, preds,
                          title="PyTorch Row-Normalised",
                          normalize="true", cmap="Greens", ax=axs[1])
    plt.tight_layout()

    return macro_f1, preds, targets

def plot_training_history(train_losses, val_losses, val_f1s,
                          xgb_baseline=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    plot_loss_f1(train_losses, val_losses, val_f1s,
                   xgb_baseline=xgb_baseline, axs=axs)
    plt.tight_layout()

def print_final_comparison(xgb_f1, pytorch_f1):
    print(f"XGBoost  Macro F1: {xgb_f1:.4f}")
    print(f"PyTorch  Macro F1: {pytorch_f1:.4f}")
    print(f"Improvement:       {pytorch_f1 - xgb_f1:+.4f}")
    
def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost baseline. Notebook-only benchmark; not part of runtime pipeline."""
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_test, y_test)], verbose=False)
    return model

def plot_confusion_matrix(targets, preds, title="Confusion Matrix",
                          labels=TARGET_NAMES, normalize="true",
                          cmap="Blues", ax=None):
    if ax is None:
        ax = plt.gca()
    cm = confusion_matrix(targets, preds, normalize=normalize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                vmin=0.0 if normalize else None,
                vmax=1.0 if normalize else None)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return ax

def plot_loss_f1(train_losses, val_losses, val_f1s,
                 xgb_baseline=None, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(train_losses, label="Train Loss")
    axs[0].plot(val_losses, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Loss Curves")
    axs[1].plot(val_f1s, label="Validation Macro F1", color="green")
    if xgb_baseline is not None:
        axs[1].axhline(y=xgb_baseline, color="red", linestyle="--",
                       label="XGBoost F1")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Macro F1")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Validation Macro F1")
    return axs