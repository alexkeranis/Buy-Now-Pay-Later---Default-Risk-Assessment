from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "BNPL_Financial_Default_Risk_Dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TRANSFORMERS_DIR = MODELS_DIR / "transformers"

DATA_PATH = PROCESSED_DIR / "final_data.npz"
MODEL_PATH = MODELS_DIR / "best_pytorch_model.pth"
SCALER_PATH = TRANSFORMERS_DIR / "robust_scaler.pkl"
CAT_ENCODER_PATH = TRANSFORMERS_DIR / "cat_encoders.pkl"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

TARGET_COL = "Default_Risk"
ID_COL = "Customer_ID"

NUM_COLS = [
    "Age",
    "Total_BNPL_Active_Loans",
    "Total_BNPL_Debt_USD",
    "Average_Transaction_Value_USD",
    "Income_USD",
    "Credit_Score",
]
CAT_COLS = [
    "Employment_Status",
    "Late_Payment_History",
    "Shopping_Category_Most_Frequent",
]
RATIO_COL = "Debt_to_Income_Ratio"
FEATURE_COLS = NUM_COLS + [RATIO_COL]

TARGET_NAMES = ["Low", "Medium", "High"]
TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}

RANGES = {
    "Age": (18, 100),
    "Total_BNPL_Active_Loans": (0, 10),
    "Total_BNPL_Debt_USD": (0, 50000),
    "Average_Transaction_Value_USD": (0, 2000),
    "Income_USD": (5000, 150000),
    "Credit_Score": (300, 850),
}

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

N_NEIGHBORS = 5

EMB_DIM = 8
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.4

BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5