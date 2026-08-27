"""
BNPL Default Risk - Visual EDA + KNN Imputation (Mixed Data)
Protocol: 
- Impute Income_USD & Credit_Score using median of 5 nearest neighbors.
- Distance computed on: Scaled numerics + One-hot encoded categoricals.
- Excludes Customer_ID and Default_Risk from imputation matrix.
- Plots raw vs imputed distributions for validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Hard-coded file path
file_path = r"C:\Users\Alexz\BNPL_ML\data\BNPL_Financial_Default_Risk_Dataset.csv"


try:
    df = pd.read_csv(file_path)
    print(f"File loaded successfully\nShape: {df.shape}\n")
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

# Create a working copy
df_raw = df.copy()

# Define target column and identifier column
target_col = 'Default_Risk'
id_col = 'Customer_ID'

# Numeric columns (including the two with missing values)
num_cols = [
    'Age', 
    'Total_BNPL_Active_Loans', 
    'Total_BNPL_Debt_USD', 
    'Average_Transaction_Value_USD',
    'Income_USD',          # has missing
    'Credit_Score'         # has missing
]

# Categorical columns (all have zero missing values)
cat_cols = [
    'Employment_Status',
    'Late_Payment_History',
    'Shopping_Category_Most_Frequent'
]



# ------------------------------
# KNN Imputation Function (Leakage-Free)
# ------------------------------
def knn_impute_train_test(train_df, test_df, num_cols, cat_cols, n_neighbors=5):
    """
    Impute missing values in train and test sets using KNN (median).
    - Fits StandardScaler, OneHotEncoder, and KNNImputer ONLY on train set.
    - Transforms train and test sets without leakage.
    Returns imputed DataFrames (original scale for numerics, original categoricals retained).
    """
    # 1. Separate features
    X_train = train_df[num_cols + cat_cols].copy()
    X_test = test_df[num_cols + cat_cols].copy()
    
    # 2. Preprocessor: scale nums, one-hot encode cats (fit on train only)
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), num_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_cols)
    ])
    preprocessor.fit(X_train)
    
    # Transform both
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 3. KNN Imputer (fit on train only)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
    imputer.fit(X_train_processed)
    
    X_train_imputed_processed = imputer.transform(X_train_processed)
    X_test_imputed_processed = imputer.transform(X_test_processed)
    
    # 4. Extract numeric columns (first len(num_cols) columns after transformation)
    n_num_cols = len(num_cols)
    train_num_scaled = X_train_imputed_processed[:, :n_num_cols]
    test_num_scaled = X_test_imputed_processed[:, :n_num_cols]
    
    # Inverse scale back to original units using the training scaler
    scaler = preprocessor.named_transformers_['scale']
    train_num_imputed = scaler.inverse_transform(train_num_scaled)
    test_num_imputed = scaler.inverse_transform(test_num_scaled)
    
    # 5. Build imputed DataFrames (keep all original columns, replace numerics)
    train_imputed = train_df.copy()
    test_imputed = test_df.copy()
    
    for i, col in enumerate(num_cols):
        train_imputed[col] = train_num_imputed[:, i]
        test_imputed[col] = test_num_imputed[:, i]
    
    return train_imputed, test_imputed



# ------------------------------
# Dataset overview, missing values, duplicates & garbage
# ------------------------------
print("="*60)
print("Dataset Overview")
print("="*60)
df_raw.info()

print("\n" + "="*60)
print("Data Quality Check:")
print("Missing values, duplicates & garbage values")
print("="*60)
total_rows = len(df_raw)
complete_rows = df_raw.dropna().shape[0]
rows_with_missing = total_rows - complete_rows
print(f"Total samples (rows):        {total_rows}")
print(f"Complete rows (no missing):  {complete_rows}")
print(f"Rows with missing values:    {rows_with_missing}")
print(f"Percentage complete:         {complete_rows/total_rows*100:.2f}%")

missing_counts = df_raw.isnull().sum()
missing_percent = (missing_counts / total_rows) * 100
missing_table = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent
})
print("\nColumns with missing values:")
print(missing_table[missing_table['Missing Count'] > 0])
print('\n')

duplicate_rows = df_raw.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows}")

print("Garbage values (e.g., negative/out‑of‑range):")

garbage_checks = {}
for col in num_cols:
    if col == 'Age':
        invalid = (df_raw[col] < 18) | (df_raw[col] > 100)
    elif col == 'Credit_Score':
        invalid = (df_raw[col] < 0) | (df_raw[col] > 1000)
    elif col in ['Income_USD', 'Total_BNPL_Debt_USD', 'Average_Transaction_Value_USD', 'Total_BNPL_Active_Loans']:
        invalid = (df_raw[col] < 0)
    else:
        invalid = (df_raw[col] < 0)
    count = invalid.sum()
    garbage_checks[col] = count
    print(f"  {col}: {count} invalid values")

inf_counts = df_raw[num_cols].isin([np.inf, -np.inf]).sum().sum()
if inf_counts > 0:
    print(f"  Warning: {inf_counts} infinite values found.")
print()

# Plotting histograms
print("="*60)
print("Numerical EDA")
print("="*60)

# Set style
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot histograms for all numeric columns (raw)
for i, col in enumerate(num_cols):
    ax = axes[i]
    # Drop NaNs for histogram
    data_clean = df_raw[col].dropna() 
    kde_flag = False if col == 'Total_BNPL_Active_Loans' else True
    sns.histplot(data_clean, kde=kde_flag, ax=ax, color='steelblue', bins=30)
    ax.set_title(f'{col} (raw)\n(non-null: {len(data_clean)})')
    ax.set_xlabel('')
    
plt.tight_layout()
plt.show()

# Boxplots comparing numeric data to target (default risk)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    # Filter out rows where target or col is NaN
    plot_df = df_raw[[col, target_col]].dropna()
    sns.boxplot(x=target_col, y=col, data=plot_df, ax=ax, hue=target_col, palette='Set2', legend=False)
    ax.set_title(f'{col} vs Default_Risk (raw)')
    ax.set_xlabel('')
    
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Categorical EDA")
print("="*60)

# Exploratory data analysis for categorical data
print("\nCategorical breakdown vs Default_Risk (as %):")
risk_order = ['Low', 'Medium', 'High']
for col in cat_cols:
    crosstab = pd.crosstab(df_raw[col], df_raw[target_col], normalize='index') * 100
    crosstab = crosstab[risk_order]     # low to high instead alphabetically
    print(f"\n--- {col} ---")
    print(crosstab.round(1))
print()

# KNN Imputation Pipeline
print("="*60)
print("Performing KNN imputation on corrupted data (k=5, median)")
print("="*60)

# Standardize all numeric columns (using non-null mean/std)
scaler_params = {}
X_scaled_list = []

for col in num_cols:
    mean = df_raw[col].mean(skipna=True)
    std = df_raw[col].std(skipna=True)
    scaler_params[col] = {'mean': mean, 'std': std}
    # Apply scaling; NaNs remain NaN
    scaled_col = (df_raw[col] - mean) / std
    X_scaled_list.append(scaled_col.values.reshape(-1, 1))

# One-hot encode categorical columns (drop first to avoid dummy trap)
cat_dummies = pd.get_dummies(df_raw[cat_cols], drop_first=True)

# Combine scaled nums + dummies into a single numpy array
X_combined = np.concatenate(X_scaled_list, axis=1)          # shape: (n, 6)
X_combined = np.concatenate([X_combined, cat_dummies.values], axis=1)  # shape: (n, 6 + n_cats)

print(f"Feature matrix shape for KNN: {X_combined.shape}")

# Run KNN Imputer (strategy='median')
imputer = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')
X_imputed_scaled = imputer.fit_transform(X_combined)

missing_before = np.isnan(X_combined).sum().sum()
missing_after = np.isnan(X_imputed_scaled).sum().sum()
print(f"KNN Imputation complete. Missing values before: {missing_before} | Missing values after: {missing_after} (should be 0)")

# Extract the imputed columns for Income_USD and Credit_Score
# They are at indices 4 and 5 in the scaled array (since num_cols order is fixed)
income_idx = num_cols.index('Income_USD')   # 4
credit_idx = num_cols.index('Credit_Score') # 5

# Inverse transform back to original scale
imputed_income_scaled = X_imputed_scaled[:, income_idx]
imputed_credit_scaled = X_imputed_scaled[:, credit_idx]

imputed_income = (imputed_income_scaled * scaler_params['Income_USD']['std']) + scaler_params['Income_USD']['mean']
imputed_credit = (imputed_credit_scaled * scaler_params['Credit_Score']['std']) + scaler_params['Credit_Score']['mean']

# Create a new DataFrame with imputed values
df_imputed = df_raw.copy()
df_imputed['Income_USD'] = imputed_income
df_imputed['Credit_Score'] = imputed_credit

# Statistics of imputed data compared to raw
print("\n" + "="*60)
print("Verifying Imputed Data Statistically")
print("="*60)
for col in ['Income_USD', 'Credit_Score']:
    orig_mean = df_raw[col].mean(skipna=True)
    orig_median = df_raw[col].median(skipna=True)
    new_mean = df_imputed[col].mean()
    new_median = df_imputed[col].median()
    
    print(f"\n--- {col} ---")
    print(f"  Original (non-null) Mean:   {orig_mean:.2f}  |  Median: {orig_median:.2f}")
    print(f"  Imputed (full) Mean:        {new_mean:.2f}  |  Median: {new_median:.2f}")
    print(f"  Difference in Mean:         {new_mean - orig_mean:.2f}")
    
# Feature Engineering: Debt_to_Income_Ratio
print("\n" + "="*60)
print("Feature Engineering: Debt_to_Income_Ratio")
print("="*60)

# Avoid division by zero (shouldn't happen, but safe)
df_imputed['Debt_to_Income_Ratio'] = df_imputed['Total_BNPL_Debt_USD'] / df_imputed['Income_USD'].replace(0, np.nan)
# Extend the numeric list for scaling (KNN imputation stays with original 6)
num_cols_with_ratio = num_cols + ['Debt_to_Income_Ratio']
print(f"Added 'Debt_to_Income_Ratio'\nUpdated numeric feature list: {num_cols_with_ratio}")
print()

# Visualizing imputed data
print("\n" + "="*60)
print("Verifying Data Imputation Graphically")
print("="*60)
print()

# Histograms of imputed vs raw (overlay)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Income_USD
ax = axes[0]
sns.histplot(df_raw['Income_USD'].dropna(), kde=True, color='red', label='Raw (non-null)', ax=ax, alpha=0.5, bins=30)
sns.histplot(df_imputed['Income_USD'], kde=True, color='blue', label='Imputed (full)', ax=ax, alpha=0.5, bins=30)
ax.set_title('Income_USD: Raw vs Imputed')
ax.legend()

# Credit_Score
ax = axes[1]
sns.histplot(df_raw['Credit_Score'].dropna(), kde=True, color='red', label='Raw (non-null)', ax=ax, alpha=0.5, bins=30)
sns.histplot(df_imputed['Credit_Score'], kde=True, color='blue', label='Imputed (full)', ax=ax, alpha=0.5, bins=30)
ax.set_title('Credit_Score: Raw vs Imputed')
ax.legend()

plt.tight_layout()
plt.show()

# Feature correlation heatmap (including imputed data)
plt.figure(figsize=(8, 6))
corr_matrix = df_imputed[num_cols_with_ratio].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap (Imputed Numeric Features)')
plt.tight_layout()
plt.show()

# Data Normalization verification for EDA
print("="*60)
print("Normalization Verification (EDA pipeline)")
print("="*60)

X = df_imputed.drop(columns=[id_col, target_col])
X_numeric = X[num_cols_with_ratio]
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Verify
scaled_df = pd.DataFrame(X_scaled, columns=num_cols_with_ratio)
summary = scaled_df.describe().round(4)
# Extract median and compute IQR
median = summary.loc['50%']
iqr = summary.loc['75%'] - summary.loc['25%']
verification_df = pd.DataFrame({'median': median, 'IQR': iqr}).T
print("Verification (should have median ≈ 0, IQR ≈ 1):")
print()
with pd.option_context('display.show_dimensions', False):
    print(verification_df.round(4))
print("\nNormalization verified for EDA purposes\n")

print("\n" + "="*60)
print("Stratified Train/Test Split & Leakage Prevention")
print("="*60)

# Stratified Split (preserve target distribution)
y = df_raw[target_col]
train_raw, test_raw = train_test_split(
    df_raw,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training set shape: {train_raw.shape}")
print(f"Test set shape:     {test_raw.shape}")
print("\nTraining target distribution:\n", train_raw[target_col].value_counts(normalize=True).round(3))
print("\nTest target distribution:\n", test_raw[target_col].value_counts(normalize=True).round(3))

# Apply KNN Imputation (leakage‑free)
train_imp, test_imp = knn_impute_train_test(train_raw, test_raw, num_cols, cat_cols)

# Add Debt_to_Income_Ratio (after imputation)
train_imp['Debt_to_Income_Ratio'] = train_imp['Total_BNPL_Debt_USD'] / train_imp['Income_USD'].replace(0, np.nan)
test_imp['Debt_to_Income_Ratio'] = test_imp['Total_BNPL_Debt_USD'] / test_imp['Income_USD'].replace(0, np.nan)

# RobustScaler (fit on train, transform both)
feature_cols = num_cols_with_ratio   # defined earlier (includes the new ratio)
scaler_model = RobustScaler()
X_train_scaled = scaler_model.fit_transform(train_imp[feature_cols])
X_test_scaled = scaler_model.transform(test_imp[feature_cols])

# Extract targets
y_train = train_imp[target_col].values
y_test = test_imp[target_col].values

print("\n" + "="*60)
print("Creating leakage-free datasets for model training and validation")
print("="*60)
print(f"X_train shape: {X_train_scaled.shape}  |  y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test_scaled.shape}  |  y_test shape:  {y_test.shape}")

print("\nScalers and imputer were fit ONLY on training data")
print("Data is now fully prepared and ready for ML model training")

# ------------------------------
# SAVE PREPROCESSED DATA FOR DOWNSTREAM USE
# ------------------------------
print("\n" + "="*60)
print("Saving processed datasets to cache")
print("="*60)

# Extract and encode categoricals for PyTorch (consistent mapping across train/test)
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

# Create folder if it doesn't exist
import os
os.makedirs('data/processed', exist_ok=True)

# Save everything
np.savez('data/processed/final_data.npz',
         X_train_scaled=X_train_scaled,
         X_test_scaled=X_test_scaled,
         y_train=y_train,
         y_test=y_test,
         cat_train=cat_train,
         cat_test=cat_test,
         cat_cardinalities=np.array(cat_cardinalities),
         cat_cols=np.array(cat_cols),
         num_cols=np.array(num_cols),
         target_col=target_col)

print("Data saved to: data/processed/final_data.npz")
print(f"   - X_train_scaled: {X_train_scaled.shape}")
print(f"   - X_test_scaled:  {X_test_scaled.shape}")
print(f"   - cat_train:      {cat_train.shape}")
print(f"   - cat_test:       {cat_test.shape}")
print(f"   - cat_cardinalities: {cat_cardinalities}")