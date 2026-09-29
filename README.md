# BNPL Default Risk Prediction

Classifies Buy Now Pay Later customers into Low, Medium, or High default risk.

## Results

The PyTorch model reached a macro F1 of 0.84 on the test set. The XGBoost baseline reached 0.75. The neural network shows a significant improvement of 9 percentage points.

## Structure

- `src/` data preparation, training, and inference scripts
- `notebooks/` EDA, model training, and an interactive demo
- `models/` saved weights and transformers
- `data/` raw and processed datasets

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch from the official index first, then run the requirements file.

## Usage

Prepare the data, then train, then predict:

```bash
python src/data_preparation.py
python src/train.py
```

For a single interactive prediction:

```bash
python src/inference.py
```

For batch predictions from a CSV:

```bash
python src/inference.py --csv path/to/customers.csv
```

The batch output is written next to the input file with `_with_predictions` appended to the name.

## Dataset

The dataset used in this project is available on Kaggle:

https://www.kaggle.com/datasets/itzzomkar/buy-now-pay-later-bnpl-default-risk/data

## License

MIT. See `LICENSE`.
```