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

# ============================================================
# 0. CONFIGURATION
# ============================================================
# Raw data path – relative to project root (since script is in src/)
RAW_DATA_PATH = "data/BNPL_Financial_Default_Risk_Dataset.csv"

# Column definitions
target_col = 'Default_Risk'
id_col = 'Customer_ID'
num_cols = ['Age','Total_BNPL_Active_Loans','Total_BNPL_Debt_USD',
            'Average_Transaction_Value_USD','Income_USD','Credit_Score']
cat_cols = ['Employment_Status','Late_Payment_History','Shopping_Category_Most_Frequent']

# ============================================================
# 1. LOAD RAW DATA
# ============================================================
print("="*60)
print("STEP 1: Loading raw data")
print("="*60)
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Raw data not found at: {RAW_DATA_PATH}")

df = pd.read_csv(RAW_DATA_PATH)
df_raw = df.copy()
print(f"Loaded shape: {df_raw.shape}")
print("First 2 rows:\n", df_raw.head(2))
print("\nColumn names and data types:")
print(df_raw.dtypes)
print()

# ============================================================
# 2. QUALITY CHECKS
# ============================================================
print("="*60)
print("STEP 2: Quality checks")
print("="*60)
print("Dataset Overview:")
df_raw.info()
print("\nMissing values per column:")
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
    count = invalid.sum()
    print(f"  {col}: {count} invalid values")
    if count > 0:
        print(f"    Example invalid values: {df_raw[col][invalid].head(3).tolist()}")
print()

# ============================================================
# 3. STRATIFIED TRAIN/TEST SPLIT
# ============================================================
print("="*60)
print("STEP 3: Stratified train/test split (80/20)")
print("="*60)
y = df_raw[target_col]
train_raw, test_raw = train_test_split(
    df_raw, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {train_raw.shape}")
print(f"Test shape:  {test_raw.shape}")
print("\nTrain target distribution:")
print(train_raw[target_col].value_counts(normalize=True).round(3))
print("\nTest target distribution:")
print(test_raw[target_col].value_counts(normalize=True).round(3))
print()

# ============================================================
# 4. KNN IMPUTATION (LEAKAGE-FREE)
# ============================================================
print("="*60)
print("STEP 4: Leakage‑free KNN imputation (k=5, median)")
print("="*60)

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

# Verify no NaNs remain in numeric columns
nan_train = train_imp[num_cols].isnull().sum().sum()
nan_test  = test_imp[num_cols].isnull().sum().sum()
print(f"NaNs remaining in train numeric cols: {nan_train} (should be 0)")
print(f"NaNs remaining in test numeric cols:  {nan_test} (should be 0)")

print("\nImputed numeric columns – summary statistics (train):")
print(train_imp[num_cols].describe().round(2))
print()

# ============================================================
# 5. ADD DEBT-TO-INCOME RATIO
# ============================================================
print("="*60)
print("STEP 5: Feature engineering – Debt_to_Income_Ratio")
print("="*60)
train_imp['Debt_to_Income_Ratio'] = train_imp['Total_BNPL_Debt_USD'] / train_imp['Income_USD'].replace(0, np.nan)
test_imp['Debt_to_Income_Ratio'] = test_imp['Total_BNPL_Debt_USD'] / test_imp['Income_USD'].replace(0, np.nan)
num_cols_with_ratio = num_cols + ['Debt_to_Income_Ratio']

print(f"Added 'Debt_to_Income_Ratio' – now {len(num_cols_with_ratio)} numeric features.")
print("Train ratio stats:\n", train_imp['Debt_to_Income_Ratio'].describe().round(3))
print("Test ratio stats:\n", test_imp['Debt_to_Income_Ratio'].describe().round(3))
print()

# ============================================================
# 6. ENCODE CATEGORICALS FOR PYTORCH
# ============================================================
print("="*60)
print("STEP 6: Encode categoricals to ordinal indices (for PyTorch embeddings)")
print("="*60)
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

print("Categorical cardinalities:")
for col, card in zip(cat_cols, cat_cardinalities):
    print(f"  {col}: {card}")
print(f"cat_train shape: {cat_train.shape}")
print(f"cat_test shape:  {cat_test.shape}")
print("First 5 rows of encoded cat_train:\n", cat_train[:5])
print()

# ============================================================
# 7. SCALING WITH ROBUSTSCALER
# ============================================================
print("="*60)
print("STEP 7: Robust scaling (fit on train, transform both)")
print("="*60)
feature_cols = num_cols_with_ratio
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_imp[feature_cols])
X_test_scaled = scaler.transform(test_imp[feature_cols])

# Verify scaling properties
scaled_df_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
scaled_df_test  = pd.DataFrame(X_test_scaled,  columns=feature_cols)
print("Train scaled – median and IQR (should be 0 and 1):")
print("  Medians:\n", scaled_df_train.median().round(4))
print("  IQR (75%-25%):\n", (scaled_df_train.quantile(0.75) - scaled_df_train.quantile(0.25)).round(4))
print()

# ============================================================
# 8. SAVE PROCESSED DATA AND TRANSFORMERS
# ============================================================
print("="*60)
print("STEP 8: Saving processed arrays and transformers")
print("="*60)

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/transformers', exist_ok=True)

save_path_npz = 'data/processed/final_data.npz'
np.savez(save_path_npz,
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
with open('models/transformers/robust_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

mappings = {}
for col in cat_cols:
    # We need the mapping from the combined encoder used above
    # The variable 'combined' is from the loop – but it holds the last column's combined.
    # We should recompute mappings for each column.
    combined_cat = pd.concat([train_imp[col], test_imp[col]], axis=0)
    encoder = pd.Categorical(combined_cat)
    mappings[col] = {cat: code for code, cat in enumerate(encoder.categories)}
with open('models/transformers/cat_encoders.pkl', 'wb') as f:
    pickle.dump(mappings, f)

# Verify saved files exist
print(f"NPZ saved to: {save_path_npz}")
print(f"  File size: {os.path.getsize(save_path_npz) / 1024:.1f} KB")
print("Transformers saved to: models/transformers/")
print("  - robust_scaler.pkl")
print("  - cat_encoders.pkl")

print("\n✅ Data preparation complete. All files are ready.")