import os
import pandas as pd
import matplotlib.pyplot as plt

# Try importing seaborn but keep a safe fallback to matplotlib only
try:
	import seaborn as sns
except Exception:
	sns = None


def find_input_file():
	candidates = [
		"Unemployment.csv",
		"Unemployment in India.csv",
		"Unemployment_Rate_upto_11_2020.csv",
	]
	for c in candidates:
		if os.path.exists(c):
			return c
	# fallback: pick the first csv in the current working directory
	for f in os.listdir('.'):
		if f.lower().endswith('.csv'):
			return f
	return None


input_file = find_input_file()
if input_file is None:
	raise FileNotFoundError("No CSV file found. Place your CSV in this folder and try again.")

# Load dataset
df = pd.read_csv(input_file)

print(f"\nLoaded file: {input_file}")

# Normalize column names: strip whitespace, convert % -> 'percent', remove parens,
# replace non-alphanumerics with underscore, collapse multiple underscores.
import re

def _normalize_col(s: str) -> str:
	s = str(s).strip()
	s = s.replace('%', ' percent ')
	s = s.replace('(', ' ').replace(')', ' ')
	s = s.replace('/', ' ')
	s = re.sub(r'[^0-9A-Za-z]+', '_', s)
	s = re.sub(r'_+', '_', s)
	s = s.strip('_').lower()
	return s

orig_columns = list(df.columns)
new_columns = [_normalize_col(c) for c in orig_columns]
df.columns = new_columns

print("\n===== Dataset Head =====")
print(df.head())

print("\n===== Dataset Columns (normalized) =====")
print(new_columns)

print("\n===== Dataset Info =====")
print(df.info())

print("\n===== Summary Statistics =====")
print(df.describe(include='all'))

# Handle missing values (if any) - keep simple approach
df = df.dropna()

# Determine unemployment-like numeric column to plot using normalized names
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
preferred_names = [
	'estimated_unemployment_rate_percent',
	'estimated_unemployment_rate',
	'unemployment_rate',
	'estimated_unemployed_rate',
	'estimated_employed',
	'estimated_labour_participation_rate_percent',
	'estimated_labour_participation_rate',
]
unemployment_col = None
for name in preferred_names:
	if name in df.columns:
		unemployment_col = name
		break
if unemployment_col is None and numeric_cols:
	unemployment_col = numeric_cols[0]

if unemployment_col is None:
	print("No numeric column found to plot distribution. Exiting.")
else:
	# Distribution plot
	plt.figure(figsize=(10, 5))
	if sns is not None:
		sns.histplot(df[unemployment_col], kde=True)
	else:
		plt.hist(df[unemployment_col].dropna(), bins=30)
	plt.title(f'Distribution of {unemployment_col}')
	plt.xlabel(unemployment_col)
	plt.ylabel('Frequency')
	plt.show()

	# Unemployment by Region if possible (use normalized 'region')
	if 'region' in df.columns:
		plt.figure(figsize=(12, 6))
		if sns is not None:
			sns.barplot(x='region', y=unemployment_col, data=df)
		else:
			grp = df.groupby('region')[unemployment_col].mean().sort_values(ascending=False)
			grp.plot(kind='bar')
		plt.xticks(rotation=45)
		plt.title(f'{unemployment_col} by region')
		plt.show()

	# Scatter with 'Estimated Employed' if present
	if 'estimated_employed' in df.columns:
		plt.figure(figsize=(8, 5))
		if sns is not None:
			sns.scatterplot(x='estimated_employed', y=unemployment_col, data=df)
		else:
			plt.scatter(df['estimated_employed'], df[unemployment_col])
		plt.title(f'Relationship Between Estimated Employed & {unemployment_col}')
		plt.xlabel('estimated_employed')
		plt.ylabel(unemployment_col)
		plt.show()

# Save cleaned dataset
out_name = "Cleaned_Unemployment.csv"
df.to_csv(out_name, index=False)
print(f"\nSaved cleaned dataset as {out_name}")
