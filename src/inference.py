# src/inference.py
import os
import sys
import numpy as np
import torch
import pickle
import argparse
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import (
    load_inference_components,
    preprocess_single_sample,
    predict_single
)

# ============================================================
# 0. CONFIGURATION AND WORKING DIRECTORY
# ============================================================
print("="*60)
print("BNPL DEFAULT RISK – INFERENCE SCRIPT")
print("="*60)
print(f"Working directory: {os.getcwd()}")

# If running from inside src/, go up one level to project root
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
    print("Changed working directory to project root:", os.getcwd())

# Define feature columns and ranges (for validation and user prompts)
num_cols = ['Age', 'Total_BNPL_Active_Loans', 'Total_BNPL_Debt_USD',
            'Average_Transaction_Value_USD', 'Income_USD', 'Credit_Score']
cat_cols = ['Employment_Status', 'Late_Payment_History', 'Shopping_Category_Most_Frequent']

ranges = {
    'Age': (18, 100),
    'Total_BNPL_Active_Loans': (0, 10),
    'Total_BNPL_Debt_USD': (0, 50000),
    'Average_Transaction_Value_USD': (0, 2000),
    'Income_USD': (5000, 150000),
    'Credit_Score': (300, 850)
}

# ============================================================
# 1. LOAD MODEL AND TRANSFORMERS
# ============================================================
print("\n" + "="*60)
print("STEP 1: Loading saved model and transformers")
print("="*60)

model_path = 'models/best_pytorch_model.pth'
scaler_path = 'models/transformers/robust_scaler.pkl'
cat_encoder_path = 'models/transformers/cat_encoders.pkl'

# Verify files exist
for path in [model_path, scaler_path, cat_encoder_path]:
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        print("Please ensure the training script has been run and saved the artifacts.")
        sys.exit(1)
    else:
        print(f"✅ Found: {path} (size: {os.path.getsize(path) / 1024:.1f} KB)")

print("All artifacts found. Loading...")
try:
    model, scaler, cat_mappings = load_inference_components(
        model_path, scaler_path, cat_encoder_path
    )
    print("✅ Model, scaler, and encoders loaded successfully.")
except Exception as e:
    print(f"❌ Error loading components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model moved to device: {device}")

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Print available categorical options
print("\nCategorical options (for reference):")
for col in cat_cols:
    options = list(cat_mappings[col].keys())
    print(f"  {col}: {options}")

# ============================================================
# 2. FUNCTIONS FOR BATCH (CSV) AND INTERACTIVE INPUT
# ============================================================
def process_single_input(input_dict):
    """Process a single dictionary of inputs."""
    # Validate numeric ranges (optional extra check)
    for col in num_cols:
        low, high = ranges[col]
        val = input_dict[col]
        if val < low or val > high:
            print(f"Warning: {col} = {val} is outside typical range [{low}, {high}]")
    X_num, X_cat = preprocess_single_sample(input_dict, num_cols, cat_cols, scaler, cat_mappings)
    pred_class, probs = predict_single(model, X_num, X_cat, device)
    return pred_class, probs

def get_user_input():
    """Prompt user for values and return a dictionary."""
    input_dict = {}
    print("\n" + "="*60)
    print("Enter customer details (type the value and press Enter)")
    print("For numeric fields, values must be within the shown range.")
    print("For categorical fields, choose from the listed options.")
    print("="*60)
    
    # Numeric fields
    for col in num_cols:
        low, high = ranges[col]
        while True:
            try:
                val = input(f"{col} [{low}-{high}]: ")
                if val.strip() == '':
                    raise ValueError("Value cannot be empty.")
                num_val = float(val)
                if num_val < low or num_val > high:
                    print(f"  Value out of range. Please enter between {low} and {high}.")
                    continue
                input_dict[col] = num_val
                break
            except ValueError as e:
                print(f"  Invalid input: {e}. Please enter a number.")
    
    # Categorical fields
    for col in cat_cols:
        options = cat_mappings[col].keys()
        print(f"\n{col} – options: {list(options)}")
        while True:
            val = input(f"{col}: ")
            if val.strip() == '':
                print("  Value cannot be empty.")
                continue
            if val in options:
                input_dict[col] = val
                break
            else:
                print(f"  '{val}' not in options. Please choose from: {list(options)}")
    return input_dict

def display_prediction(pred_class, probs):
    target_names = ['Low', 'Medium', 'High']
    pred_label = target_names[pred_class]
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Predicted Default Risk: {pred_label}")
    print("\nClass Probabilities:")
    for label, prob in zip(target_names, probs):
        print(f"  {label}: {prob*100:.2f}%")
    print("="*60)

# ============================================================
# 3. BATCH INFERENCE FROM CSV (if argument provided)
# ============================================================
def batch_inference(csv_path):
    """Read CSV, preprocess, predict, and save results."""
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    print("Columns:", df.columns.tolist())
    
    # Check required columns exist
    required = num_cols + cat_cols
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return
    
    # Process each row
    predictions = []
    probabilities = []
    print("Processing rows...")
    for idx, row in df.iterrows():
        input_dict = {col: row[col] for col in required}
        # Convert numerics to float
        for col in num_cols:
            try:
                input_dict[col] = float(input_dict[col])
            except (ValueError, TypeError):
                print(f"  Warning: Row {idx}, column {col} has non-numeric value. Setting to 0.")
                input_dict[col] = 0.0
        # Categoricals are strings; if NaN, replace with first category
        for col in cat_cols:
            if pd.isna(input_dict[col]):
                print(f"  Warning: Row {idx}, column {col} is missing. Using default category.")
                input_dict[col] = list(cat_mappings[col].keys())[0]
        pred_class, probs = process_single_input(input_dict)
        predictions.append(pred_class)
        probabilities.append(probs)
    
    # Add results to DataFrame
    target_names = ['Low', 'Medium', 'High']
    df['Predicted_Risk'] = [target_names[p] for p in predictions]
    probs_df = pd.DataFrame(probabilities, columns=[f'Prob_{c}' for c in target_names])
    df = pd.concat([df, probs_df], axis=1)
    
    # Save
    output_path = csv_path.replace('.csv', '_with_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")
    print("\nPreview of predictions:")
    print(df[['Predicted_Risk'] + [f'Prob_{c}' for c in target_names]].head())

# ============================================================
# 4. MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BNPL Default Risk Inference')
    parser.add_argument('--csv', type=str, help='Path to CSV file for batch inference')
    parser.add_argument('--test', action='store_true', help='Run a test prediction with hardcoded values')
    args = parser.parse_args()
    
    if args.csv:
        batch_inference(args.csv)
    elif args.test:
        print("\nRunning test with a sample customer.")
        test_input = {
            'Age': 35,
            'Total_BNPL_Active_Loans': 2,
            'Total_BNPL_Debt_USD': 5000,
            'Average_Transaction_Value_USD': 150,
            'Income_USD': 40000,
            'Credit_Score': 650,
            'Employment_Status': 'Employed',
            'Late_Payment_History': 'No',
            'Shopping_Category_Most_Frequent': 'Fashion'
        }
        print("Test input:", test_input)
        pred_class, probs = process_single_input(test_input)
        display_prediction(pred_class, probs)
    else:
        print("\nRunning in interactive mode.")
        while True:
            user_data = get_user_input()
            pred_class, probs = process_single_input(user_data)
            display_prediction(pred_class, probs)
            
            again = input("\nMake another prediction? (y/n): ").strip().lower()
            if again not in ['y', 'yes']:
                print("Exiting inference.")
                break