# Task 3 — Car Price Prediction (Enhanced)

This folder contains an enhanced version of the car price prediction script that produces visualizations and evaluates a linear regression model.

How to run:

1. Create and activate a Python environment (recommended):
```powershell
python -m venv .venv; .\.venv\Scripts\Activate
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. Run the script (plot images will be saved to a `plots/` directory):
```powershell
python car_price_prediction.py
```

What this script does:
- Loads `car_data.csv` (this file should be in the same folder)
- Preprocesses data (one-hot encoding of categorical variables)
- Trains a LinearRegression model
- Creates and saves visualizations:
  - Histogram of Selling Price and Present Price
  - Correlation heatmap
  - Scatter plots of selected features against Selling Price (with regression line)
  - Boxplots for categorical variables vs Selling Price
  - Residual diagnostics
  - Feature importances (absolute coefficients)

Tips:
- If you prefer interactive plots, convert `plt.show()` to interactive backends or use Plotly/Streamlit.
- Try using other regressors (e.g. RandomForestRegressor) to evaluate improvements in explained variance.
