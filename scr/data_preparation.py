# src/data_prep.py
import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Hard-coded raw data path (adjust if needed)
RAW_DATA_PATH = "../data/BNPL_Financial_Default_Risk_Dataset.csv"

# --- 1. Load raw data ---
df = pd.read_csv(RAW_DATA_PATH)
df_raw = df.copy()

target_col = 'Default_Risk'
id_col = 'Customer_ID'
num_cols = ['Age','Total_BNPL_Active_Loans','Total_BNPL_Debt_USD',
            'Average_Transaction_Value_USD','Income_USD','Credit_Score']
cat_cols = ['Employment_Status','Late_Payment_History','Shopping_Category_Most_Frequent']

# --- 2. Quality checks (prints only) ---
print("="*60)
print("Dataset Overview")
df_raw.info()
print("\nMissing values:")
print(df_raw.isnull().sum()[df_raw.isnull().sum()>0])
print(f"\nDuplicate rows: {df_raw.duplicated().sum()}")
print("\nGarbage values (negative/out-of-range):")
for col in num_cols:
    if col == 'Age':
        invalid = (df_raw[col] < 18) | (df_raw[col] > 100)
    elif col == 'Credit_Score':
        invalid = (df_raw[col] < 0) | (df_raw[col] > 1000)
    else:
        invalid = (df_raw[col] < 0)
    print(f"  {col}: {invalid.sum()}")

# --- 3. Stratified split ---
y = df_raw[target_col]
train_raw, test_raw = train_test_split(
    df_raw, test_size=0.2, random_state=42, stratify=y
)

# --- 4. KNN Imputation function (leakage-free) ---
def knn_impute_train_test(train_df, test_df, num_cols, cat_cols, n_neighbors=5):
    X_train = train_df[num_cols + cat_cols].copy()
    X_test = test_df[num_cols + cat_cols].copy()
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), num_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_cols)
    ])
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
    imputer.fit(X_train_proc)
    X_train_imp = imputer.transform(X_train_proc)
    X_test_imp = imputer.transform(X_test_proc)
    # Extract numeric part and inverse transform
    n_num = len(num_cols)
    scaler = preprocessor.named_transformers_['scale']
    train_num = scaler.inverse_transform(X_train_imp[:, :n_num])
    test_num = scaler.inverse_transform(X_test_imp[:, :n_num])
    # Build DataFrames
    train_imputed = train_df.copy()
    test_imputed = test_df.copy()
    for i, col in enumerate(num_cols):
        train_imputed[col] = train_num[:, i]
        test_imputed[col] = test_num[:, i]
    return train_imputed, test_imputed

train_imp, test_imp = knn_impute_train_test(train_raw, test_raw, num_cols, cat_cols)

# --- 5. Add Debt_to_Income_Ratio ---
train_imp['Debt_to_Income_Ratio'] = train_imp['Total_BNPL_Debt_USD'] / train_imp['Income_USD'].replace(0, np.nan)
test_imp['Debt_to_Income_Ratio'] = test_imp['Total_BNPL_Debt_USD'] / test_imp['Income_USD'].replace(0, np.nan)
num_cols_with_ratio = num_cols + ['Debt_to_Income_Ratio']

# --- 6. Prepare categorical indices for PyTorch (consistent mapping) ---
cat_train_raw = train_imp[cat_cols].copy()
cat_test_raw = test_imp[cat_cols].copy()
for col in cat_cols:
    combined = pd.concat([cat_train_raw[col], cat_test_raw[col]], axis=0)
    encoder = pd.Categorical(combined)
    cat_train_raw[col] = encoder.codes[:len(cat_train_raw)]
    cat_test_raw[col] = encoder.codes[len(cat_train_raw):]

cat_train = cat_train_raw.values
cat_test = cat_test_raw.values
cat_cardinalities = [train_imp[col].nunique() for col in cat_cols]

# --- 7. Scaling with RobustScaler (fit on train only) ---
feature_cols = num_cols_with_ratio
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_imp[feature_cols])
X_test_scaled = scaler.transform(test_imp[feature_cols])

# --- 8. Save processed arrays and transformers ---
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models/transformers', exist_ok=True)

np.savez('../data/processed/final_data.npz',
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

# Save transformers
with open('../models/transformers/robust_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('../models/transformers/cat_encoders.pkl', 'wb') as f:
    # save the category mapping for each column
    mappings = {}
    for col in cat_cols:
        mappings[col] = {cat: code for code, cat in enumerate(combined.categories)}  # combined from above
    pickle.dump(mappings, f)

print("Data preparation complete. Cache saved to data/processed/final_data.npz")
print("Transformers saved to models/transformers/")