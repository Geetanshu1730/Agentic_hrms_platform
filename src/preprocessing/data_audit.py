import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"


FILES = [
    "attrition_clean.csv",
    "performance_engagement_clean.csv",
    "occupations_clean.csv",
    "essential_skills_clean.csv",
    "software_skills_clean.csv"
]


# ============================================================
# DATASET AUDIT FUNCTION
# ============================================================

def audit_dataset(filename):

    path = DATA_DIR / filename

    print("\n")
    print("=" * 90)
    print(f"DATASET: {filename}")
    print("=" * 90)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(path)

    print("\n1. SHAPE")
    print("-" * 40)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # COLUMN INFORMATION
    # --------------------------------------------------------

    print("\n2. COLUMN INFORMATION")
    print("-" * 40)

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing": df.isnull().sum().values,
        "Unique": df.nunique().values
    })

    print(info.to_string(index=False))

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    print("\n3. DUPLICATES")
    print("-" * 40)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    print("\n4. MISSING VALUES")
    print("-" * 40)

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        for column, count in missing.items():

            percentage = (count / len(df)) * 100

            print(
                f"{column}: "
                f"{count} missing "
                f"({percentage:.2f}%)"
            )

    # --------------------------------------------------------
    # CATEGORICAL COLUMNS
    # --------------------------------------------------------

    print("\n5. CATEGORICAL COLUMNS")
    print("-" * 40)

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_columns) == 0:

        print("No categorical columns.")

    else:

        for column in categorical_columns:

            print(f"\n{column}")

            print(
                df[column]
                .value_counts(dropna=False)
                .head(15)
                .to_string()
            )

    # --------------------------------------------------------
    # NUMERICAL COLUMNS
    # --------------------------------------------------------

    print("\n6. NUMERICAL COLUMNS")
    print("-" * 40)

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    if len(numerical_columns) == 0:

        print("No numerical columns.")

    else:

        print(
            df[numerical_columns]
            .describe()
            .round(2)
            .to_string()
        )

    # --------------------------------------------------------
    # SAMPLE DATA
    # --------------------------------------------------------

    print("\n7. FIRST 5 ROWS")
    print("-" * 40)

    print(
        df.head()
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    print("\n8. MEMORY USAGE")
    print("-" * 40)

    memory = df.memory_usage(
        deep=True
    ).sum() / (1024 ** 2)

    print(f"{memory:.2f} MB")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 90)
    print("             AI WORKFORCE INTELLIGENCE")
    print("                    DATA AUDIT")
    print("=" * 90)

    for filename in FILES:

        try:

            audit_dataset(filename)

        except FileNotFoundError:

            print(
                f"\nERROR: {filename} was not found."
            )

        except Exception as e:

            print(
                f"\nERROR processing {filename}: {e}"
            )

    print("\n")
    print("=" * 90)
    print("                    AUDIT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()