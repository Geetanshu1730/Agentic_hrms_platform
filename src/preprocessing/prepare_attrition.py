import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "raw" / "employee_attrition.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "attrition_clean.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading employee attrition dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# --------------------------------------------------
# REMOVE UNNECESSARY COLUMNS
# --------------------------------------------------

columns_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# --------------------------------------------------
# CONVERT TARGET
# --------------------------------------------------

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# --------------------------------------------------
# CHECK TARGET
# --------------------------------------------------

print("\nAttrition distribution:")

print(
    df["Attrition"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")

print(
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      .head(10)
)


# --------------------------------------------------
# SAVE CLEAN DATA
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# FINAL INFORMATION
# --------------------------------------------------

print("\nFinal shape:", df.shape)

print("\nClean dataset saved to:")

print(OUTPUT_FILE)

print("\nPreprocessing complete!")