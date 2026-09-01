import pandas as pd
from pathlib import Path


# Project root
ROOT = Path(__file__).resolve().parents[2]

# Dataset location
DATA_DIR = ROOT / "data" / "raw"


FILES = [
    "employee_attrition.csv",
    "hr_performance_engagement.csv",
    "occupation_data.csv",
    "essential_skills.csv",
    "software_skills.csv",
]


def inspect_dataset(filename):
    path = DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"DATASET: {filename}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: File not found -> {path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()

    for column, count in missing.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"  - {column}: {count} ({percentage:.2f}%)")

    if missing.sum() == 0:
        print("  No missing values found.")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nUnique values for categorical columns:")

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:
        unique_count = df[column].nunique()

        print(f"\n  {column}")
        print(f"  Unique values: {unique_count}")

        if unique_count <= 20:
            print(f"  Values: {df[column].dropna().unique().tolist()}")
        else:
            print("  Too many values to display.")

    print("\nNumerical summary:")

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    if len(numerical_columns) > 0:
        print(df[numerical_columns].describe().round(2).to_string())
    else:
        print("  No numerical columns found.")


def main():

    print("\n")
    print("=" * 80)
    print("        ENTERPRISE HR AI — DATA INSPECTION")
    print("=" * 80)

    for filename in FILES:
        inspect_dataset(filename)

    print("\n")
    print("=" * 80)
    print("DATA INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()