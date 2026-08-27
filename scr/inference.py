# src/inference.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
sys.path.append(os.path.dirname(__file__))
from utils import EmbeddingNet, encode_target

def load_transformers(path='../models/transformers/'):
    with open(os.path.join(path, 'robust_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(path, 'cat_encoders.pkl'), 'rb') as f:
        cat_mappings = pickle.load(f)
    return scaler, cat_mappings

def preprocess_new_data(df, num_cols, cat_cols, scaler, cat_mappings):
    # Ensure numeric columns are present and scale them
    X_num_raw = df[num_cols].copy()
    X_num_scaled = scaler.transform(X_num_raw)
    # Encode categoricals using mappings
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        mapping = cat_mappings[col]
        X_cat[:, i] = df[col].map(lambda x: mapping.get(x, 0)).values  # default to 0 if unknown
    return X_num_scaled, X_cat

def load_model(model_path='../models/best_pytorch_model.pth',
               num_numeric=7, cat_cardinalities=[4,2,5]):
    model = EmbeddingNet(num_numeric=num_numeric, cat_cardinalities=cat_cardinalities)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, X_num, X_cat, device='cpu'):
    with torch.no_grad():
        X_num = torch.tensor(X_num, dtype=torch.float32).to(device)
        X_cat = torch.tensor(X_cat, dtype=torch.long).to(device)
        outputs = model(X_num, X_cat)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

if __name__ == "__main__":
    # Example: load a CSV file (must have same columns as original)
    # Adjust the path to your new data file.
    new_data_path = "../data/new_customers.csv"  # example
    if not os.path.exists(new_data_path):
        print(f"Sample file not found: {new_data_path}")
        print("Please provide a CSV with columns: " + 
              "Age, Employment_Status, Income_USD, Credit_Score, Total_BNPL_Active_Loans, "
              "Total_BNPL_Debt_USD, Late_Payment_History, Shopping_Category_Most_Frequent, Average_Transaction_Value_USD")
        sys.exit(1)

    df_new = pd.read_csv(new_data_path)
    # Define column lists (must match training)
    num_cols = ['Age','Total_BNPL_Active_Loans','Total_BNPL_Debt_USD',
                'Average_Transaction_Value_USD','Income_USD','Credit_Score']
    cat_cols = ['Employment_Status','Late_Payment_History','Shopping_Category_Most_Frequent']

    # Load transformers and model
    scaler, cat_mappings = load_transformers()
    # We need cat_cardinalities; we can load from the saved model or hardcode.
    # For simplicity, we load from the same prepared data or hardcode.
    # We'll assume we have a separate file or we load from final_data.npz.
    # Let's load them from the cache:
    from utils import load_processed_data
    data = load_processed_data()
    cat_cardinalities = data['cat_cardinalities']

    model = load_model(num_numeric=len(num_cols), cat_cardinalities=cat_cardinalities)

    # Preprocess
    X_num, X_cat = preprocess_new_data(df_new, num_cols, cat_cols, scaler, cat_mappings)

    # Predict
    preds, probs = predict(model, X_num, X_cat, device='cpu')

    # Map predictions back to labels
    inv_mapping = {0:'Low', 1:'Medium', 2:'High'}
    pred_labels = [inv_mapping[p] for p in preds]

    # Add to DataFrame and save
    df_new['Predicted_Default_Risk'] = pred_labels
    df_new[['Prob_Low','Prob_Medium','Prob_High']] = probs
    output_path = "../data/predictions.csv"
    df_new.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    print(df_new[['Predicted_Default_Risk','Prob_Medium','Prob_High']].head())