# -----------------------------------------------------------
# TASK 5 - SALES PREDICTION USING MACHINE LEARNING (OIBSIP)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = pd.read_csv("advertising.csv")   # change file name if different

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== DESCRIPTION =====")
print(df.describe())

# -----------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------
sns.pairplot(df)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------------------------------------
# FEATURES & TARGET
# -----------------------------------------------------------
X = df[['TV', 'Radio', 'Newspaper']]   # independent variables
y = df['Sales']                         # dependent variable

# -----------------------------------------------------------
# TRAIN-TEST SPLIT
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------
# MODEL EVALUATION
# -----------------------------------------------------------
y_pred = model.predict(X_test)

print("\n===== MODEL COEFFICIENTS =====")
print(model.coef_)

print("\n===== MEAN SQUARED ERROR =====")
print(mean_squared_error(y_test, y_pred))

print("\n===== R2 SCORE =====")
print(r2_score(y_test, y_pred))

# -----------------------------------------------------------
# PLOT PREDICTION VS ACTUAL
# -----------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

print("\n===== TASK 5 COMPLETED SUCCESSFULLY =====")

# -----------------------------------------------------------
# EXAMPLE PREDICTION
# -----------------------------------------------------------

sample_input = pd.DataFrame({
    'TV': [150],
    'Radio': [20],
    'Newspaper': [15]
})

sample_pred = model.predict(sample_input)

print("\nSample Prediction for:")
print(sample_input)
print("Predicted Sales:", sample_pred[0])
