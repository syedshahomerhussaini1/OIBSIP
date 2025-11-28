# Task 4 — Spam Email Detection (Enhanced)

This folder contains an enhanced Spam detection script with more meaningful visualisations and analysis.

How to run:

1. Create and activate a Python environment (recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. Run the script:
```powershell
python spam_detection.py
```

Outputs:
- Saved plots under `plots/`: class distribution, confusion matrix, ROC curve, PR curve, top features, and wordclouds
- Saved model under `models/` (joblib)
- CSV file with misclassified examples (`plots/misclassified_samples.csv`)

Notes:
- The script uses a Logistic Regression with TF-IDF features as a simple baseline. Try tree-based or ensemble models for improvements.
- Wordclouds and top-word visualizations give interpretability for spam/ham lexical differences.
