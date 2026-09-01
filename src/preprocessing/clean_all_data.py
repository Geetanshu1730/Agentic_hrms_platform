import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GENERAL CLEANING FUNCTION
# ============================================================

def basic_clean(df):

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    # Strip whitespace from text columns
    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    return df


# ============================================================
# 1. EMPLOYEE ATTRITION
# ============================================================

def clean_attrition():

    print("\n" + "=" * 70)
    print("1. CLEANING EMPLOYEE ATTRITION")
    print("=" * 70)

    path = RAW_DIR / "employee_attrition.csv"

    df = pd.read_csv(path)

    print("Original:", df.shape)

    # Basic cleaning
    df = basic_clean(df)

    # Remove constant / unnecessary columns
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

    # Convert target
    if "Attrition" in df.columns:

        df["Attrition"] = (
            df["Attrition"]
            .map({
                "Yes": 1,
                "No": 0
            })
        )

    # Fill numerical missing values with median
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    for column in numerical_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Fill categorical missing values with mode
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        if df[column].isnull().any():

            mode = df[column].mode()

            if len(mode) > 0:
                df[column] = df[column].fillna(
                    mode[0]
                )

    output = PROCESSED_DIR / "attrition_clean.csv"

    df.to_csv(
        output,
        index=False
    )

    print("Final:", df.shape)
    print("Saved:", output)


# ============================================================
# 2. PERFORMANCE & ENGAGEMENT
# ============================================================

def clean_performance():

    print("\n" + "=" * 70)
    print("2. CLEANING PERFORMANCE & ENGAGEMENT")
    print("=" * 70)

    path = RAW_DIR / "hr_performance_engagement.csv"

    df = pd.read_csv(path)

    print("Original:", df.shape)

    df = basic_clean(df)

    # Fill numerical missing values
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    for column in numerical_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Fill categorical missing values
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        if df[column].isnull().any():

            mode = df[column].mode()

            if len(mode) > 0:
                df[column] = df[column].fillna(
                    mode[0]
                )

    output = (
        PROCESSED_DIR
        / "performance_engagement_clean.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("Final:", df.shape)
    print("Saved:", output)


# ============================================================
# 3. OCCUPATION DATA
# ============================================================

def clean_occupation():

    print("\n" + "=" * 70)
    print("3. CLEANING OCCUPATION DATA")
    print("=" * 70)

    path = RAW_DIR / "occupation_data.csv"

    df = pd.read_csv(path)

    print("Original:", df.shape)

    df = basic_clean(df)

    # Normalize text
    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Replace string "nan" with actual missing value
    df = df.replace(
        ["nan", "None", ""],
        np.nan
    )

    # Remove rows where every value is missing
    df = df.dropna(
        how="all"
    )

    output = (
        PROCESSED_DIR
        / "occupations_clean.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("Final:", df.shape)
    print("Saved:", output)


# ============================================================
# 4. ESSENTIAL SKILLS
# ============================================================

def clean_essential_skills():

    print("\n" + "=" * 70)
    print("4. CLEANING ESSENTIAL SKILLS")
    print("=" * 70)

    path = RAW_DIR / "essential_skills.csv"

    df = pd.read_csv(path)

    print("Original:", df.shape)

    df = basic_clean(df)

    # Normalize text fields
    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Replace artificial missing strings
    df = df.replace(
        ["nan", "None", ""],
        np.nan
    )

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    )

    # Remove duplicates
    df = df.drop_duplicates()

    output = (
        PROCESSED_DIR
        / "essential_skills_clean.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("Final:", df.shape)
    print("Saved:", output)


# ============================================================
# 5. SOFTWARE SKILLS
# ============================================================

def clean_software_skills():

    print("\n" + "=" * 70)
    print("5. CLEANING SOFTWARE SKILLS")
    print("=" * 70)

    path = RAW_DIR / "software_skills.csv"

    df = pd.read_csv(path)

    print("Original:", df.shape)

    df = basic_clean(df)

    # Normalize text
    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Replace artificial missing values
    df = df.replace(
        ["nan", "None", ""],
        np.nan
    )

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    )

    # Remove duplicates
    df = df.drop_duplicates()

    output = (
        PROCESSED_DIR
        / "software_skills_clean.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("Final:", df.shape)
    print("Saved:", output)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("       AI WORKFORCE INTELLIGENCE PLATFORM")
    print("              DATA CLEANING PIPELINE")
    print("=" * 70)

    clean_attrition()
    clean_performance()
    clean_occupation()
    clean_essential_skills()
    clean_software_skills()

    print("\n")
    print("=" * 70)
    print("           ALL DATASETS CLEANED")
    print("=" * 70)


if __name__ == "__main__":
    main()