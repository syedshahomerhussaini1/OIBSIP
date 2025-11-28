import argparse
import os
import sys

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from csv path and return pandas DataFrame.

    Tries to find file directly and relative to script if not absolute.
    """
    # Resolve path: absolute or relative to script
    if not os.path.isabs(csv_path):
        basedir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(basedir, csv_path)

    # If path points to a directory (e.g., you have a folder named 'Iris.csv' with the CSV file inside),
    # try to find a CSV file inside it.
    if os.path.isdir(csv_path):
        found = None
        for root, _, files in os.walk(csv_path):
            for f in files:
                if f.lower().endswith(".csv"):
                    found = os.path.join(root, f)
                    break
            if found:
                break
        if found:
            csv_path = found
        else:
            raise FileNotFoundError(f"No CSV files found inside directory: {csv_path}")

    # If path doesn't exist at all, search recursively within the script directory for a matching filename
    if not os.path.exists(csv_path):
        basedir = os.path.dirname(os.path.abspath(__file__))
        found = None
        base_name = os.path.basename(csv_path)
        for root, _, files in os.walk(basedir):
            if base_name in files:
                found = os.path.join(root, base_name)
                break
        if found:
            csv_path = found
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def prepare_features(df: pd.DataFrame):
    # Drop Id column if present
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)

    if "Species" not in df.columns:
        raise ValueError("Target column 'Species' not found in DataFrame")

    X = df.drop("Species", axis=1)
    y = df["Species"]
    return X, y


def train_and_evaluate(X, y, plot=True, sample=None):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    if sample is not None:
        try:
            sample_scaled = scaler.transform(sample)
            prediction = model.predict(sample_scaled)
            print("Prediction for sample input:", prediction[0])
        except Exception as e:
            print("Could not predict sample input:", e)

    if plot:
        try:
            # If original column names are available, use them by reconstructing DataFrame
            if hasattr(X, "columns"):
                plot_df = pd.concat([pd.DataFrame(X), y.reset_index(drop=True)], axis=1)
            else:
                plot_df = pd.concat([pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]), y.reset_index(drop=True)], axis=1)
            sns.pairplot(plot_df, hue="Species")
            plt.show()
        except Exception:
            # Fallback: attempt to plot without hue
            try:
                sns.pairplot(plot_df)
                plt.show()
            except Exception as ex:
                print("Could not generate pairplot:", ex)

    return model, scaler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate a logistic regression on the Iris dataset")
    parser.add_argument("csv", nargs="?", default="Iris.csv", help="Path to the Iris CSV file (default: Iris.csv in script dir)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting (useful in headless environments)")
    parser.add_argument("--sample", nargs=4, type=float, metavar=("sepal_length", "sepal_width", "petal_length", "petal_width"),
                        help="Give a sample 4-value input to classify")

    args = parser.parse_args(argv)

    try:
        df = load_data(args.csv)
    except Exception as e:
        print(e)
        sys.exit(1)

    X, y = prepare_features(df)

    sample = None
    if args.sample is not None:
        sample = [args.sample]

    train_and_evaluate(X, y, plot=not args.no_plot, sample=sample)


if __name__ == "__main__":
    main()
