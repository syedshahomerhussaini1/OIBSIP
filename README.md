
# Iris Classification

This repository contains a Python script that trains a Logistic Regression model on the Iris dataset and prints evaluation metrics.

Files:
- `iris_classification.py`: Main script (correct spelling) with CLI support.
- `iris_classificaton.py`: Original file — contains the same logic but misspelled filename.
- `Iris.csv/Iris.csv`: The dataset - keep in the same directory as the script.
- `requirements.txt`: Python dependencies.

Quick start (PowerShell):

```powershell
# 1. Create and activate virtual env (optional but recommended)
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Install requirements
pip install -r requirements.txt

# 3. Run script
python iris_classification.py

# 4. (Optional) Run without plotting (useful in headless/server env)
python iris_classification.py --no-plot

# 5. (Optional) Provide a custom CSV path and sample
python iris_classification.py path\to\Iris.csv --sample 5.1 3.5 1.4 0.2
```

Notes:
- The script will search for `Iris.csv` relative to the script directory by default.
  If your dataset is inside a folder named `Iris.csv` (for example: `Iris.csv\Iris.csv`), the script
  will automatically search for a CSV file inside that directory and use it. If you still see a `PermissionError`
  like "[Errno 13] Permission denied: '...\Iris.csv'", that usually means the path passed points to a directory
  instead of a file. Possible fixes:
  - Move `Iris.csv` file to the root of the repository (as `Iris.csv`).
  - Pass the correct file path explicitly:
    ```powershell
    python iris_classification.py .\Iris.csv\Iris.csv
    ```
  - Rename the folder so the script doesn't confuse a directory for a file.
- If you prefer, you can run the existing `iris_classificaton.py` file (typo) instead:
  `python iris_classificaton.py`.
