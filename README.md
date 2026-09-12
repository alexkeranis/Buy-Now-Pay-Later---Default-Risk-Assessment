# BNPL Default Risk Prediction

End‑to‑end ML pipeline to classify BNPL customers into Low/Medium/High default risk.

## Key Results
- **XGBoost baseline**: Macro F1 = 0.771  
- **PyTorch with Entity Embeddings + Weighted CrossEntropy**: Macro F1 = **0.845**  
- **Improvement**: +7.4%

## Project Structure
- `src/` – Python modules (data preparation, training, inference)
- `notebooks/` – detailed and visual EDA, model building/comparison, interactive predictions
- `models/` – saved model weights and transformers
- `data/` – raw and processed data

## Setup
```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt