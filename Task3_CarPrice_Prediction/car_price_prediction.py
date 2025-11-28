
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure plotting style
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 5)


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def preprocess_df(df: pd.DataFrame, target_col: str = "Selling_Price") -> pd.DataFrame:
    # Copy to avoid side effects
    df = df.copy()

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Fill missing numeric values (if any) with median
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Treat categorical variables using one-hot encoding for interpretability
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in categorical_cols if c != target_col]
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred


def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return dict(r2=r2, mae=mae, mse=mse, rmse=rmse)


def ensure_plots_dir(out_dir: str = "plots"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_histograms(df: pd.DataFrame, out_dir="plots"):
    # Plot histogram for Selling Price and Present Price (if present)
    if "Selling_Price" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Selling_Price"], bins=30, kde=True)
        plt.title("Distribution of Selling Price")
        plt.xlabel("Selling Price (in lakhs)")
        fname = os.path.join(out_dir, "hist_selling_price.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.show()

    for col in ["Present_Price"]:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            fname = os.path.join(out_dir, f"hist_{col}.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.show()


def plot_correlation_matrix(df: pd.DataFrame, out_dir="plots"):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Matrix (upper triangle)")
    fname = os.path.join(out_dir, "correlation_heatmap.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.show()


def plot_scatter_features_vs_target(df: pd.DataFrame, target_col: str = "Selling_Price", out_dir="plots"):
    # Choose a few numeric predictors that are likely important
    candidate_cols = ["Present_Price", "Driven_kms", "Year"]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in candidate_cols:
        if c in df.columns and c in numeric_cols and c != target_col:
            plt.figure(figsize=(8, 5))
            sns.regplot(x=df[c], y=df[target_col], scatter_kws={"s": 20}, line_kws={"color": "red"}, ci=None)
            plt.title(f"{c} vs {target_col}")
            plt.xlabel(c)
            plt.ylabel(target_col)
            fname = os.path.join(out_dir, f"scatter_{c}_vs_{target_col}.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.show()


def plot_boxplots_for_categoricals(original_df: pd.DataFrame, target_col: str = "Selling_Price", out_dir="plots"):
    # Use original un-encoded frame to create categorical plots
    df = original_df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col == target_col:
            continue
        # If the categorical variable has too many levels, only show top categories
        nunique = df[col].nunique()
        if nunique > 20:
            top_categories = df[col].value_counts().nlargest(10).index.tolist()
            df_plot = df[df[col].isin(top_categories)].copy()
            title = f"{target_col} by {col} (top 10 categories)"
        else:
            df_plot = df
            title = f"{target_col} by {col}"

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y=target_col, data=df_plot)
        plt.xticks(rotation=45)
        plt.title(title)
        fname = os.path.join(out_dir, f"box_{col}_vs_{target_col}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.show()


def plot_predictions_vs_actual(y_test, y_pred, out_dir="plots"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Selling Price")
    fname = os.path.join(out_dir, "actual_vs_predicted.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.show()


def plot_residuals(y_test, y_pred, out_dir="plots"):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    fname = os.path.join(out_dir, "residuals_hist.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    fname = os.path.join(out_dir, "residuals_vs_predicted.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, X_train: pd.DataFrame, out_dir="plots"):
    # Linear regression coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_
        feat_importances = pd.Series(coefs, index=X_train.columns)
        feat_importances = feat_importances.abs().sort_values(ascending=False)[:20]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feat_importances.values, y=feat_importances.index)
        plt.title("Feature importances (|coef|) - top 20")
        fname = os.path.join(out_dir, "feature_importances.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.show()


def main():
    # Paths
    csv_path = "car_data.csv"
    out_dir = ensure_plots_dir("plots")

    # Load and inspect
    df = load_data(csv_path)
    print("\n===== SAMPLE =====")
    print(df.head())
    print("\n===== INFO =====")
    print(df.info())
    print("\n===== SUMMARY =====")
    print(df.describe())

    # Keep a copy of original for categorical plots
    orig_df = df.copy()

    # Preprocess
    df_pre = preprocess_df(df, target_col="Selling_Price")

    # Split features and target
    if "Selling_Price" not in df_pre.columns:
        raise KeyError("Selling_Price column not found in data after preprocessing")
    X = df_pre.drop("Selling_Price", axis=1)
    y = df_pre["Selling_Price"]

    # Train model
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred)
    print("\n===== MODEL PERFORMANCE =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Visualizations
    plot_histograms(orig_df)
    plot_correlation_matrix(df_pre, out_dir)
    plot_scatter_features_vs_target(orig_df, target_col="Selling_Price", out_dir=out_dir)
    plot_boxplots_for_categoricals(orig_df, target_col="Selling_Price", out_dir=out_dir)
    plot_predictions_vs_actual(y_test, y_pred, out_dir)
    plot_residuals(y_test, y_pred, out_dir)
    plot_feature_importance(model, X_train, out_dir)

    print("\n===== TASK COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
# ----------------------------------------
# TASK 3 - CAR PRICE PREDICTION (OIBSIP)
# ----------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("car_data.csv")  # change name if needed

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== SUMMARY STATS =====")
print(df.describe())

# ----------------------------
# HANDLE CATEGORICAL FEATURES
# ----------------------------
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = label_encoder.fit_transform(df[col])

print("\n===== DATA AFTER ENCODING =====")
print(df.head())

# ----------------------------
# SEPARATE FEATURES & TARGET
# ----------------------------
X = df.drop("Selling_Price", axis=1)  # change column name if needed
y = df["Selling_Price"]

# ----------------------------
# TRAIN–TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# PREDICTIONS
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# MODEL EVALUATION
# ----------------------------
print("\n===== MODEL PERFORMANCE =====")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ----------------------------
# VISUALIZATIONS
# ----------------------------

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\n===== TASK 3 COMPLETED SUCCESSFULLY =====")
