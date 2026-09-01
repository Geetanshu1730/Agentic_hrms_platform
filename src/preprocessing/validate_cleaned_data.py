import pandas as pd
import numpy as np
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
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print("\n")
    print("=" * 90)
    print(title)
    print("=" * 90)


def subsection(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# GENERAL VALIDATION
# ============================================================

def validate_general(df, filename):

    subsection("GENERAL VALIDATION")

    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Duplicates : {df.duplicated().sum()}")
    print(f"Missing    : {df.isnull().sum().sum()}")

    # Infinite values
    numeric_df = df.select_dtypes(include=np.number)

    if len(numeric_df.columns) > 0:
        infinite_count = np.isinf(numeric_df).sum().sum()
        print(f"Infinite values : {infinite_count}")
    else:
        print("Infinite values : N/A")


# ============================================================
# ATTRITION VALIDATION
# ============================================================

def validate_attrition(df):

    section("ATTRITION DATASET VALIDATION")

    subsection("TARGET DISTRIBUTION")

    if "Attrition" in df.columns:

        counts = df["Attrition"].value_counts().sort_index()

        print(counts.to_string())

        percentages = (
            df["Attrition"]
            .value_counts(normalize=True)
            .sort_index() * 100
        )

        print("\nPercentage distribution:")
        print(percentages.round(2).to_string())

        if len(counts) == 2:
            minority_percentage = percentages.min()

            if minority_percentage < 20:
                print(
                    "\nWARNING: Attrition classes are imbalanced."
                )
            else:
                print(
                    "\nClass distribution looks acceptable."
                )

    subsection("CATEGORICAL FEATURES")

    categorical = df.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns

    for column in categorical:

        print(
            f"{column}: "
            f"{df[column].nunique()} unique values"
        )

    subsection("POSSIBLE IDENTIFIER COLUMNS")

    identifier_keywords = [
        "id",
        "number",
        "code"
    ]

    possible_ids = []

    for column in df.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in identifier_keywords
        ):
            possible_ids.append(column)

    if possible_ids:
        print("Possible identifier columns:")
        for column in possible_ids:
            print(f"  - {column}")
    else:
        print("No obvious identifier columns found.")

    subsection("VALUE RANGE CHECKS")

    range_checks = {
        "Age": (18, 100),
        "JobLevel": (1, 5),
        "JobSatisfaction": (1, 4),
        "EnvironmentSatisfaction": (1, 4),
        "WorkLifeBalance": (1, 4),
        "PerformanceRating": (1, 5),
        "PercentSalaryHike": (0, 100),
        "TotalWorkingYears": (0, 60),
        "YearsAtCompany": (0, 60),
    }

    for column, (minimum, maximum) in range_checks.items():

        if column in df.columns:

            invalid = df[
                (df[column] < minimum) |
                (df[column] > maximum)
            ]

            print(
                f"{column}: "
                f"{len(invalid)} invalid values"
            )


# ============================================================
# PERFORMANCE VALIDATION
# ============================================================

def validate_performance(df):

    section("PERFORMANCE & ENGAGEMENT VALIDATION")

    subsection("IDENTIFIER / PERSONAL COLUMNS")

    possible_identifiers = [
        "Unnamed:_0",
        "FirstName",
        "LastName",
        "ADEmail",
        "Employee_ID",
        "DOB",
        "Supervisor"
    ]

    for column in possible_identifiers:

        if column in df.columns:
            print(f"  {column}")

    subsection("PERFORMANCE DISTRIBUTION")

    if "Performance_Score" in df.columns:

        print(
            df["Performance_Score"]
            .value_counts(dropna=False)
            .to_string()
        )

    subsection("EMPLOYEE STATUS")

    if "EmployeeStatus" in df.columns:

        print(
            df["EmployeeStatus"]
            .value_counts(dropna=False)
            .to_string()
        )

    subsection("EMPLOYEE TYPE")

    if "EmployeeType" in df.columns:

        print(
            df["EmployeeType"]
            .value_counts(dropna=False)
            .to_string()
        )

    subsection("SCORE RANGE CHECKS")

    score_columns = [
        "Current_Employee_Rating",
        "Engagement_Score",
        "Satisfaction_Score",
        "Work-Life_Balance_Score",
        "Training_Duration(Days)"
    ]

    for column in score_columns:

        if column in df.columns:

            invalid = df[
                (df[column] < 1) |
                (df[column] > 5)
            ]

            print(
                f"{column}: "
                f"{len(invalid)} invalid values"
            )


# ============================================================
# OCCUPATION VALIDATION
# ============================================================

def validate_occupations(df):

    section("OCCUPATION DATASET VALIDATION")

    subsection("STRUCTURE")

    print(
        f"Unique occupation codes : "
        f"{df['O*NET-SOC_Code'].nunique()}"
    )

    print(
        f"Unique titles : "
        f"{df['Title'].nunique()}"
    )

    print(
        f"Unique descriptions : "
        f"{df['Description'].nunique()}"
    )

    subsection("EMPTY TEXT CHECK")

    for column in [
        "O*NET-SOC_Code",
        "Title",
        "Description"
    ]:

        empty = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(
            f"{column}: {empty} empty values"
        )


# ============================================================
# ESSENTIAL SKILLS VALIDATION
# ============================================================

def validate_essential_skills(df):

    section("ESSENTIAL SKILLS VALIDATION")

    subsection("SCALE DISTRIBUTION")

    if "Scale_Name" in df.columns:

        print(
            df["Scale_Name"]
            .value_counts()
            .to_string()
        )

    subsection("SKILL DISTRIBUTION")

    print(
        f"Unique occupations : "
        f"{df['O*NET-SOC_Code'].nunique()}"
    )

    print(
        f"Unique skills : "
        f"{df['Element_Name'].nunique()}"
    )

    subsection("NOT RELEVANT")

    if "Not_Relevant" in df.columns:

        print(
            df["Not_Relevant"]
            .value_counts(dropna=False)
            .to_string()
        )

    subsection("DATA VALUE RANGE")

    if "Data_Value" in df.columns:

        print(
            f"Minimum : {df['Data_Value'].min()}"
        )

        print(
            f"Maximum : {df['Data_Value'].max()}"
        )

        invalid = df[
            (df["Data_Value"] < 0) |
            (df["Data_Value"] > 10)
        ]

        print(
            f"Invalid Data_Value rows : "
            f"{len(invalid)}"
        )


# ============================================================
# SOFTWARE SKILLS VALIDATION
# ============================================================

def validate_software_skills(df):

    section("SOFTWARE SKILLS VALIDATION")

    subsection("STRUCTURE")

    print(
        f"Unique occupations : "
        f"{df['O*NET-SOC_Code'].nunique()}"
    )

    print(
        f"Unique software examples : "
        f"{df['Workplace_Example'].nunique()}"
    )

    print(
        f"Unique software categories : "
        f"{df['Element_Name'].nunique()}"
    )

    subsection("HOT TECHNOLOGY")

    if "Hot_Technology" in df.columns:

        print(
            df["Hot_Technology"]
            .value_counts()
            .to_string()
        )

    subsection("IN DEMAND")

    if "In_Demand" in df.columns:

        print(
            df["In_Demand"]
            .value_counts()
            .to_string()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "AI WORKFORCE INTELLIGENCE PLATFORM\n"
        "STEP 5A — CLEANED DATA VALIDATION"
    )

    for filename in FILES:

        path = DATA_DIR / filename

        if not path.exists():

            print(
                f"\nERROR: {filename} not found."
            )

            continue

        try:

            df = pd.read_csv(path)

            section(f"DATASET: {filename}")

            validate_general(df, filename)

            if filename == "attrition_clean.csv":

                validate_attrition(df)

            elif filename == "performance_engagement_clean.csv":

                validate_performance(df)

            elif filename == "occupations_clean.csv":

                validate_occupations(df)

            elif filename == "essential_skills_clean.csv":

                validate_essential_skills(df)

            elif filename == "software_skills_clean.csv":

                validate_software_skills(df)

        except Exception as e:

            print(
                f"\nERROR processing {filename}: {e}"
            )

    section(
        "STEP 5A VALIDATION COMPLETE"
    )

    print(
        "\nIf no serious warnings appear, "
        "the datasets are ready for feature engineering."
    )


if __name__ == "__main__":
    main()
    