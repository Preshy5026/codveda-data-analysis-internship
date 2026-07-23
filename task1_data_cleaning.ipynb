"""
Task 1: Data Cleaning and Preprocessing
Dataset: Iris dataset (raw/messy version simulating a real-world export)
Tools: Python, pandas
"""

import pandas as pd

# ---------------------------------------------------------
# 1. Load the dataset using pandas
# ---------------------------------------------------------
df = pd.read_csv("iris_raw.csv")

print("=" * 50)
print("STEP 1: Initial dataset overview")
print("=" * 50)
print(f"Shape: {df.shape}")
print(df.info())
print(df.head())

# ---------------------------------------------------------
# 2. Identify and handle missing values
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 2: Missing values")
print("=" * 50)
missing_before = df.isna().sum()
print(missing_before)

# Numeric columns: impute with the column median (robust to outliers,
# better than mean for skewed measurement data).
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

print("\nMissing values after imputation:")
print(df.isna().sum())

# ---------------------------------------------------------
# 3. Remove duplicate rows and standardize inconsistent formats
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 3: Duplicates")
print("=" * 50)
dupes_count = df.duplicated().sum()
print(f"Duplicate rows found: {dupes_count}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape}")

print("\n" + "=" * 50)
print("STEP 4: Standardizing categorical formatting")
print("=" * 50)
print("Unique species values before standardizing:")
print(df["species"].unique())

# Standardize text case (VIRGINICA / Virginica / virginica -> virginica)
df["species"] = df["species"].str.strip().str.lower()

print("\nUnique species values after standardizing:")
print(df["species"].unique())

# ---------------------------------------------------------
# 5. Final check + export
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 5: Final cleaned dataset")
print("=" * 50)
print(f"Final shape: {df.shape}")
print(df.describe())

df.to_csv("iris_cleaned.csv", index=False)
print("\nSaved cleaned dataset to iris_cleaned.csv")
