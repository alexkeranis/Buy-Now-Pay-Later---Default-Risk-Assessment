import argparse
import sys
import numpy as np
import pandas as pd
import torch

from config import (
    CAT_COLS, CAT_ENCODER_PATH, FEATURE_COLS, MODEL_CONFIG_PATH, MODEL_PATH,
    NUM_COLS, RANGES, SCALER_PATH, TARGET_NAMES,
)
from utils import (
    load_inference_components, predict_single, preprocess_single_sample,
)


def process_single_input(input_dict, model, scaler, cat_mappings, device):
    d = dict(input_dict)
    income = float(d["Income_USD"])
    debt = float(d["Total_BNPL_Debt_USD"])
    d["Debt_to_Income_Ratio"] = debt / income if income != 0 else 0.0

    for col in NUM_COLS:
        low, high = RANGES[col]
        val = d[col]
        if val < low or val > high:
            print(f"Warning: {col} = {val} outside typical range [{low}, {high}]")

    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
        if expected != FEATURE_COLS:
            raise ValueError(
                f"Column mismatch between scaler and inference script:\n"
                f"  scaler fit on : {expected}\n"
                f"  script passing: {FEATURE_COLS}"
            )

    X_num, X_cat = preprocess_single_sample(
        d, FEATURE_COLS, CAT_COLS, scaler, cat_mappings
    )
    return predict_single(model, X_num, X_cat, device)


def get_user_input(cat_mappings):
    input_dict = {}
    print("Enter customer details for default risk prediction.")

    for col in NUM_COLS:
        low, high = RANGES[col]
        while True:
            raw = input(f"{col} [{low}-{high}]: ").strip()
            if not raw:
                print("Empty input.")
                continue
            try:
                val = float(raw)
            except ValueError:
                print("Not a number.")
                continue
            if val < low or val > high:
                print(f"Out of range. Enter between {low} and {high}.")
                continue
            input_dict[col] = val
            break

    for col in CAT_COLS:
        options = list(cat_mappings[col].keys())
        print(f"\n{col} options: {options}")
        while True:
            val = input(f"{col}: ").strip()
            if val in options:
                input_dict[col] = val
                break
            print(f"'{val}' not in options. Choose from: {options}")
    return input_dict


def display_prediction(input_dict, pred_class, probs):
    print("\nInputs:")
    for k, v in input_dict.items():
        print(f"  {k}: {v}")
    label = TARGET_NAMES[pred_class]
    print(f"\nPredicted Default Risk: {label} "
          f"(confidence: {probs[pred_class] * 100:.2f}%)")
    print("Class probabilities:")
    for name, p in zip(TARGET_NAMES, probs):
        print(f"  {name}: {p * 100:.2f}%")


def batch_inference(csv_path, model, scaler, cat_mappings, device):
    df = pd.read_csv(csv_path)
    required = NUM_COLS + CAT_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Look up the training median for each numeric column from the fitted scaler.
    # RobustScaler stores medians in center_, aligned with the columns it was fit on.
    if hasattr(scaler, "center_"):
        feature_idx = {col: FEATURE_COLS.index(col) for col in NUM_COLS}
        fallback = {col: float(scaler.center_[feature_idx[col]]) for col in NUM_COLS}
    else:
        fallback = {col: 0.0 for col in NUM_COLS}

    predictions, probabilities = [], []
    for idx, row in df.iterrows():
        d = {c: row[c] for c in required}

        for col in NUM_COLS:
            try:
                val = float(d[col])
            except (ValueError, TypeError):
                print(f"Row {idx}: {col} not numeric, using training median.")
                val = fallback[col]
            if not np.isfinite(val):
                print(f"Row {idx}: {col} missing, using training median.")
                val = fallback[col]
            d[col] = val

        for col in CAT_COLS:
            if pd.isna(d[col]):
                print(f"Row {idx}: {col} missing, using 'Unknown'.")
                d[col] = "Unknown"

        pred, probs = process_single_input(d, model, scaler, cat_mappings, device)
        predictions.append(pred)
        probabilities.append(probs)

    df["Predicted_Risk"] = [TARGET_NAMES[p] for p in predictions]
    probs_df = pd.DataFrame(probabilities, columns=[f"Prob_{c}" for c in TARGET_NAMES])
    df = pd.concat([df, probs_df], axis=1)

    out_path = csv_path.replace(".csv", "_with_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="BNPL default risk inference")
    parser.add_argument("--csv", type=str, help="CSV file for batch inference")
    args = parser.parse_args()

    for path in (MODEL_PATH, SCALER_PATH, CAT_ENCODER_PATH, MODEL_CONFIG_PATH):
        if not path.exists():
            print(f"Missing artifact: {path}. Run training first.")
            sys.exit(1)

    model, scaler, cat_mappings = load_inference_components(
        MODEL_PATH, SCALER_PATH, CAT_ENCODER_PATH, MODEL_CONFIG_PATH
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.csv:
        batch_inference(args.csv, model, scaler, cat_mappings, device)
        return

    while True:
        user_data = get_user_input(cat_mappings)
        pred, probs = process_single_input(user_data, model, scaler, cat_mappings, device)
        display_prediction(user_data, pred, probs)
        again = input("\nAnother prediction? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            break


if __name__ == "__main__":
    main()