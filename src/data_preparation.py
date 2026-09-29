import pickle

from config import CAT_ENCODER_PATH, DATA_PATH, RAW_DATA_PATH, SCALER_PATH
from utils import load_raw_data, prepare_and_save_data


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at: {RAW_DATA_PATH}")

    df = load_raw_data(RAW_DATA_PATH)

    result, scaler, cat_mappings = prepare_and_save_data(df, DATA_PATH, save=True)

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(CAT_ENCODER_PATH, "wb") as f:
        pickle.dump(cat_mappings, f)


if __name__ == "__main__":
    main()