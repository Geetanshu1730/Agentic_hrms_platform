import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = ROOT / "data" / "processed" / "features"


FILES = [
    "attrition_features.csv",
    "performance_engagement_features.csv",
    "essential_skills_features.csv",
    "software_skills_features.csv",
    "occupations_features.csv"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_section(title):
    print("\n")
    print("-" * 80)
    print(title)
    print("-" * 80)


# ============================================================
# GENERAL VALIDATION
# ============================================================

def validate_general(df):

    print_section("GENERAL VALIDATION")

    print(f"Rows       : {df.shape[0]}")
    print(f"Columns    : {df.shape[1]}")

    duplicates = df.duplicated().sum()
    missing = df.isnull().sum().sum()

    numeric_df = df.select_dtypes(include=["number"])

    if len(numeric_df.columns) > 0:
        infinite_values = np.isinf(
            numeric_df.to_numpy()
        ).sum()
    else:
        infinite_values = 0

    print(f"Duplicates : {duplicates}")
    print(f"Missing    : {missing}")
    print(f"Infinite values : {infinite_values}")

    if duplicates == 0:
        print("[OK] No duplicate rows")
    else:
        print(f"[WARNING] {duplicates} duplicate rows")

    if missing == 0:
        print("[OK] No missing values")
    else:
        print(f"[WARNING] {missing} missing values")

    if infinite_values == 0:
        print("[OK] No infinite values")
    else:
        print(
            f"[WARNING] {infinite_values} infinite values"
        )


# ============================================================
# CONSTANT COLUMN CHECK
# ============================================================

def validate_constant_columns(df):

    print_section("CONSTANT COLUMN CHECK")

    constant_columns = []

    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    if len(constant_columns) == 0:

        print("[OK] No constant columns found")

    else:

        print("[WARNING] Constant columns:")

        for column in constant_columns:
            print(f"  {column}")


# ============================================================
# NUMERICAL FEATURE VALIDATION
# ============================================================

def validate_numeric_features(df):

    print_section("NUMERICAL FEATURE CHECK")

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    if len(numeric_columns) == 0:

        print("No numerical features.")

        return

    print(
        f"Numerical features: {len(numeric_columns)}"
    )

    print("\nFeature ranges:")

    for column in numeric_columns:

        minimum = df[column].min()
        maximum = df[column].max()
        mean = df[column].mean()

        print(
            f"{column}: "
            f"min={minimum:.3f}, "
            f"max={maximum:.3f}, "
            f"mean={mean:.3f}"
        )


# ============================================================
# CATEGORICAL FEATURE VALIDATION
# ============================================================

def validate_categorical_features(df):

    print_section("CATEGORICAL FEATURE CHECK")

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns

    if len(categorical_columns) == 0:

        print("No categorical features.")

        return

    print(
        f"Categorical features: {len(categorical_columns)}"
    )

    for column in categorical_columns:

        unique_count = df[column].nunique(
            dropna=False
        )

        print(
            f"{column}: "
            f"{unique_count} unique values"
        )


# ============================================================
# ATTRITION VALIDATION
# ============================================================

def validate_attrition(df):

    print_section("ATTRITION FEATURE VALIDATION")

    if "Attrition" not in df.columns:

        print(
            "[WARNING] Attrition target not found."
        )

        return

    print("TARGET DISTRIBUTION")

    print(
        df["Attrition"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPercentage distribution:")

    print(
        (
            df["Attrition"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        )
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # Target values
    # --------------------------------------------------------

    invalid_target = ~df["Attrition"].isin([0, 1])

    print(
        f"\nInvalid Attrition values: "
        f"{invalid_target.sum()}"
    )

    if invalid_target.sum() == 0:
        print("[OK] Target contains only 0 and 1")
    else:
        print("[WARNING] Invalid target values found")

    # --------------------------------------------------------
    # New feature checks
    # --------------------------------------------------------

    print("\nENGINEERED FEATURE CHECK")

    expected_features = [
        "OverTime_Binary",
        "BusinessTravel_Level",
        "CompanyTenureRatio",
        "PromotionWaitRatio",
        "CurrentRoleTenureRatio",
        "ManagerTenureRatio",
        "Satisfaction_Index",
        "Career_Experience_Index",
        "Income_Per_JobLevel",
        "Age_Group",
        "Income_Group",
        "Risk_Flag"
    ]

    for feature in expected_features:

        if feature in df.columns:
            print(f"  [OK] {feature}")
        else:
            print(f"  [WARNING] Missing: {feature}")


# ============================================================
# PERFORMANCE VALIDATION
# ============================================================

def validate_performance(df):

    print_section(
        "PERFORMANCE & ENGAGEMENT FEATURE VALIDATION"
    )

    expected_features = [
        "Performance_Score_Numeric",
        "Engagement_Index",
        "Employee_Wellbeing_Index",
        "Training_Outcome_Numeric",
        "Training_Success",
        "Training_Passed",
        "Training_Effectiveness",
        "Training_Cost_Per_Day",
        "Performance_Engagement_Gap",
        "High_Engagement_Flag",
        "Low_Satisfaction_Flag",
        "Performance_Risk_Flag",
        "Active_Employee_Flag",
        "Training_Cost_Category",
        "Engagement_Category"
    ]

    print("ENGINEERED FEATURE CHECK")

    for feature in expected_features:

        if feature in df.columns:
            print(f"  [OK] {feature}")
        else:
            print(
                f"  [WARNING] Missing: {feature}"
            )

    # --------------------------------------------------------
    # Performance score
    # --------------------------------------------------------

    if "Performance_Score_Numeric" in df.columns:

        invalid = ~df[
            "Performance_Score_Numeric"
        ].between(1, 4)

        print(
            f"\nPerformance score invalid values: "
            f"{invalid.sum()}"
        )

    # --------------------------------------------------------
    # Engagement
    # --------------------------------------------------------

    if "Engagement_Index" in df.columns:

        invalid = ~df[
            "Engagement_Index"
        ].between(1, 5)

        print(
            f"Engagement index invalid values: "
            f"{invalid.sum()}"
        )

    # --------------------------------------------------------
    # Wellbeing
    # --------------------------------------------------------

    if "Employee_Wellbeing_Index" in df.columns:

        invalid = ~df[
            "Employee_Wellbeing_Index"
        ].between(1, 5)

        print(
            f"Wellbeing index invalid values: "
            f"{invalid.sum()}"
        )

    # --------------------------------------------------------
    # Training effectiveness
    # --------------------------------------------------------

    if "Training_Effectiveness" in df.columns:

        invalid = ~df[
            "Training_Effectiveness"
        ].between(0, 1)

        print(
            f"Training effectiveness invalid values: "
            f"{invalid.sum()}"
        )


# ============================================================
# ESSENTIAL SKILLS VALIDATION
# ============================================================

def validate_essential_skills(df):

    print_section(
        "ESSENTIAL SKILLS FEATURE VALIDATION"
    )

    expected_features = [
        "Skill_Importance",
        "Skill_Level",
        "Skill_Strength",
        "Skill_Strength_Normalized",
        "Skill_Importance_Category",
        "Skill_Level_Category"
    ]

    print("ENGINEERED FEATURE CHECK")

    for feature in expected_features:

        if feature in df.columns:
            print(f"  [OK] {feature}")
        else:
            print(
                f"  [WARNING] Missing: {feature}"
            )

    # --------------------------------------------------------
    # Skill importance
    # --------------------------------------------------------

    if "Skill_Importance" in df.columns:

        invalid = ~df[
            "Skill_Importance"
        ].between(0, 6)

        print(
            f"\nSkill importance invalid values: "
            f"{invalid.sum()}"
        )

    # --------------------------------------------------------
    # Skill level
    # --------------------------------------------------------

    if "Skill_Level" in df.columns:

        invalid = ~df[
            "Skill_Level"
        ].between(0, 6)

        print(
            f"Skill level invalid values: "
            f"{invalid.sum()}"
        )

    # --------------------------------------------------------
    # Normalized strength
    # --------------------------------------------------------

    if "Skill_Strength_Normalized" in df.columns:

        invalid = ~df[
            "Skill_Strength_Normalized"
        ].between(0, 1)

        print(
            f"Normalized strength invalid values: "
            f"{invalid.sum()}"
        )

    # --------------------------------------------------------
    # Occupations and skills
    # --------------------------------------------------------

    if "O*NET-SOC_Code" in df.columns:

        print(
            f"Unique occupations : "
            f"{df['O*NET-SOC_Code'].nunique()}"
        )

    if "Element_Name" in df.columns:

        print(
            f"Unique skills : "
            f"{df['Element_Name'].nunique()}"
        )


# ============================================================
# SOFTWARE SKILLS VALIDATION
# ============================================================

def validate_software_skills(df):

    print_section(
        "SOFTWARE SKILLS FEATURE VALIDATION"
    )

    expected_features = [
        "Hot_Technology_Binary",
        "In_Demand_Binary",
        "Technology_Demand_Score",
        "Technology_Category",
        "Software_Name",
        "Software_Category"
    ]

    print("ENGINEERED FEATURE CHECK")

    for feature in expected_features:

        if feature in df.columns:
            print(f"  [OK] {feature}")
        else:
            print(
                f"  [WARNING] Missing: {feature}"
            )

    # --------------------------------------------------------
    # Binary checks
    # --------------------------------------------------------

    for column in [
        "Hot_Technology_Binary",
        "In_Demand_Binary"
    ]:

        if column in df.columns:

            invalid = ~df[column].isin([0, 1])

            print(
                f"{column} invalid values: "
                f"{invalid.sum()}"
            )

    # --------------------------------------------------------
    # Demand score
    # --------------------------------------------------------

    if "Technology_Demand_Score" in df.columns:

        invalid = ~df[
            "Technology_Demand_Score"
        ].between(0, 2)

        print(
            f"Technology demand score "
            f"invalid values: {invalid.sum()}"
        )

    # --------------------------------------------------------
    # Technology category
    # --------------------------------------------------------

    if "Technology_Category" in df.columns:

        print("\nTechnology categories:")

        print(
            df["Technology_Category"]
            .value_counts()
            .to_string()
        )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    if "O*NET-SOC_Code" in df.columns:

        print(
            f"\nUnique occupations: "
            f"{df['O*NET-SOC_Code'].nunique()}"
        )

    if "Software_Name" in df.columns:

        print(
            f"Unique software names: "
            f"{df['Software_Name'].nunique()}"
        )


# ============================================================
# OCCUPATION VALIDATION
# ============================================================

def validate_occupations(df):

    print_section(
        "OCCUPATION FEATURE VALIDATION"
    )

    expected_features = [
        "Occupation_Title",
        "Occupation_Description",
        "Description_Length",
        "Description_Word_Count",
        "Management_Role_Flag",
        "Technology_Role_Flag"
    ]

    print("ENGINEERED FEATURE CHECK")

    for feature in expected_features:

        if feature in df.columns:
            print(f"  [OK] {feature}")
        else:
            print(
                f"  [WARNING] Missing: {feature}"
            )

    # --------------------------------------------------------
    # Description checks
    # --------------------------------------------------------

    if "Description_Length" in df.columns:

        invalid = (
            df["Description_Length"] <= 0
        )

        print(
            f"\nInvalid description lengths: "
            f"{invalid.sum()}"
        )

    if "Description_Word_Count" in df.columns:

        invalid = (
            df["Description_Word_Count"] <= 0
        )

        print(
            f"Invalid word counts: "
            f"{invalid.sum()}"
        )

    # --------------------------------------------------------
    # Binary feature checks
    # --------------------------------------------------------

    for column in [
        "Management_Role_Flag",
        "Technology_Role_Flag"
    ]:

        if column in df.columns:

            invalid = ~df[column].isin([0, 1])

            print(
                f"{column} invalid values: "
                f"{invalid.sum()}"
            )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    if "O*NET-SOC_Code" in df.columns:

        print(
            f"\nUnique occupation codes: "
            f"{df['O*NET-SOC_Code'].nunique()}"
        )

    if "Occupation_Title" in df.columns:

        print(
            f"Unique occupation titles: "
            f"{df['Occupation_Title'].nunique()}"
        )


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(filename):

    path = FEATURES_DIR / filename

    print("\n")
    print("=" * 80)
    print(f"DATASET: {filename}")
    print("=" * 80)

    if not path.exists():

        print(
            f"[ERROR] File not found: {path}"
        )

        return False

    df = pd.read_csv(path)

    # General validation
    validate_general(df)

    # Constant columns
    validate_constant_columns(df)

    # Numeric features
    validate_numeric_features(df)

    # Categorical features
    validate_categorical_features(df)

    # Dataset-specific validation
    if filename == "attrition_features.csv":

        validate_attrition(df)

    elif filename == "performance_engagement_features.csv":

        validate_performance(df)

    elif filename == "essential_skills_features.csv":

        validate_essential_skills(df)

    elif filename == "software_skills_features.csv":

        validate_software_skills(df)

    elif filename == "occupations_features.csv":

        validate_occupations(df)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("AI WORKFORCE INTELLIGENCE PLATFORM")
    print("STEP 5C - FEATURE VALIDATION")
    print("=" * 80)

    successful = 0

    for filename in FILES:

        try:

            if validate_dataset(filename):
                successful += 1

        except Exception as e:

            print(
                f"\n[ERROR] Processing {filename}: {e}"
            )

    print("\n")
    print("=" * 80)
    print("STEP 5C VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"\nDatasets validated: "
        f"{successful}/{len(FILES)}"
    )

    if successful == len(FILES):

        print(
            "\n[OK] All feature datasets "
            "successfully validated."
        )

        print(
            "\nNext step: STEP 5D - "
            "ML-READY TRANSFORMATIONS"
        )

    else:

        print(
            "\n[WARNING] Some datasets "
            "could not be validated."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()