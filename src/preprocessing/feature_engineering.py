import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "processed" / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ATTRITION FEATURE ENGINEERING
# ============================================================

def engineer_attrition_features():

    print("\n")
    print("=" * 80)
    print("STEP 5B.1 - ATTRITION FEATURE ENGINEERING")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD CLEANED DATA
    # --------------------------------------------------------

    input_path = PROCESSED_DIR / "attrition_clean.csv"

    df = pd.read_csv(input_path)

    print("\nOriginal dataset:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # CREATE COPY
    # --------------------------------------------------------

    features = df.copy()

    # --------------------------------------------------------
    # 1. OVERTIME FEATURE
    # --------------------------------------------------------

    features["OverTime_Binary"] = (
        features["OverTime"]
        .map({"Yes": 1, "No": 0})
    )

    # --------------------------------------------------------
    # 2. BUSINESS TRAVEL FEATURE
    # --------------------------------------------------------

    travel_mapping = {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2
    }

    features["BusinessTravel_Level"] = (
        features["BusinessTravel"]
        .map(travel_mapping)
    )

    # --------------------------------------------------------
    # 3. CAREER STABILITY
    # --------------------------------------------------------

    features["CompanyTenureRatio"] = (
        features["YearsAtCompany"] /
        features["TotalWorkingYears"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 4. PROMOTION WAIT
    # --------------------------------------------------------

    features["PromotionWaitRatio"] = (
        features["YearsSinceLastPromotion"] /
        features["YearsAtCompany"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 5. ROLE TENURE RATIO
    # --------------------------------------------------------

    features["CurrentRoleTenureRatio"] = (
        features["YearsInCurrentRole"] /
        features["YearsAtCompany"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 6. MANAGER TENURE RATIO
    # --------------------------------------------------------

    features["ManagerTenureRatio"] = (
        features["YearsWithCurrManager"] /
        features["YearsAtCompany"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 7. JOB SATISFACTION INDEX
    # --------------------------------------------------------

    features["Satisfaction_Index"] = (
        features["JobSatisfaction"] +
        features["EnvironmentSatisfaction"] +
        features["RelationshipSatisfaction"] +
        features["WorkLifeBalance"]
    ) / 4

    # --------------------------------------------------------
    # 8. CAREER EXPERIENCE INDEX
    # --------------------------------------------------------

    features["Career_Experience_Index"] = (
        features["TotalWorkingYears"] +
        features["YearsAtCompany"] +
        features["YearsInCurrentRole"] +
        features["YearsWithCurrManager"]
    ) / 4

    # --------------------------------------------------------
    # 9. INCOME PER JOB LEVEL
    # --------------------------------------------------------

    features["Income_Per_JobLevel"] = (
        features["MonthlyIncome"] /
        features["JobLevel"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 10. AGE GROUP
    # --------------------------------------------------------

    features["Age_Group"] = pd.cut(
        features["Age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=[
            "Young",
            "Early_Career",
            "Mid_Career",
            "Senior",
            "Late_Career"
        ]
    )

    # --------------------------------------------------------
    # 11. INCOME GROUP
    # --------------------------------------------------------

    features["Income_Group"] = pd.qcut(
        features["MonthlyIncome"],
        q=4,
        labels=[
            "Low",
            "Medium",
            "High",
            "Very_High"
        ],
        duplicates="drop"
    )

    # --------------------------------------------------------
    # 12. HIGH ATTRITION RISK INDICATOR
    # --------------------------------------------------------

    features["Risk_Flag"] = (
        (
            (features["OverTime"] == "Yes") &
            (features["JobSatisfaction"] <= 2)
        )
        |
        (
            features["YearsAtCompany"] <= 2
        )
        |
        (
            features["WorkLifeBalance"] <= 2
        )
    ).astype(int)

    # --------------------------------------------------------
    # CHECK NEW FEATURES
    # --------------------------------------------------------

    print("\nNew features created:")
    
    new_columns = [
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

    for column in new_columns:
        print(f"  [OK] {column}")

    # --------------------------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------------------------

    print("\nMissing values after feature engineering:")
    missing = features.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  [OK] No missing values")
    else:
        print(missing)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        FEATURES_DIR /
        "attrition_features.csv"
    )

    features.to_csv(
        output_path,
        index=False
    )

    print("\nOutput:")
    print(output_path)

    print("\nFinal dataset:")
    print(f"Rows    : {features.shape[0]}")
    print(f"Columns : {features.shape[1]}")

    print("\n")
    print("=" * 80)
    print("ATTRITION FEATURE ENGINEERING COMPLETE")
    print("=" * 80)


# ============================================================
# PERFORMANCE & ENGAGEMENT FEATURE ENGINEERING
# ============================================================

def engineer_performance_features():

    print("\n")
    print("=" * 80)
    print("STEP 5B.2 - PERFORMANCE & ENGAGEMENT FEATURE ENGINEERING")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD CLEANED DATA
    # --------------------------------------------------------

    input_path = PROCESSED_DIR / "performance_engagement_clean.csv"

    df = pd.read_csv(input_path)

    print("\nOriginal dataset:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # CREATE COPY
    # --------------------------------------------------------

    features = df.copy()

    # --------------------------------------------------------
    # 1. PERFORMANCE SCORE NUMERIC
    # --------------------------------------------------------

    performance_mapping = {
        "Needs Improvement": 1,
        "PIP": 2,
        "Fully Meets": 3,
        "Exceeds": 4
    }

    features["Performance_Score_Numeric"] = (
        features["Performance_Score"]
        .map(performance_mapping)
    )

    # --------------------------------------------------------
    # 2. ENGAGEMENT INDEX
    # --------------------------------------------------------

    features["Engagement_Index"] = (
        features["Engagement_Score"] +
        features["Satisfaction_Score"] +
        features["Work-Life_Balance_Score"]
    ) / 3

    # --------------------------------------------------------
    # 3. EMPLOYEE WELLBEING INDEX
    # --------------------------------------------------------

    features["Employee_Wellbeing_Index"] = (
        features["Engagement_Score"] +
        features["Satisfaction_Score"] +
        features["Work-Life_Balance_Score"] +
        features["Current_Employee_Rating"]
    ) / 4

    # --------------------------------------------------------
    # 4. TRAINING OUTCOME NUMERIC
    # --------------------------------------------------------

    training_outcome_mapping = {
        "Failed": 0,
        "Incomplete": 1,
        "Completed": 2,
        "Passed": 3
    }

    features["Training_Outcome_Numeric"] = (
        features["Training_Outcome"]
        .map(training_outcome_mapping)
    )

    # --------------------------------------------------------
    # 5. TRAINING SUCCESS FLAG
    # --------------------------------------------------------

    features["Training_Success"] = (
        features["Training_Outcome"]
        .isin(["Completed", "Passed"])
        .astype(int)
    )

    # --------------------------------------------------------
    # 6. TRAINING PASS FLAG
    # --------------------------------------------------------

    features["Training_Passed"] = (
        features["Training_Outcome"]
        .eq("Passed")
        .astype(int)
    )

    # --------------------------------------------------------
    # 7. TRAINING EFFECTIVENESS
    # --------------------------------------------------------

    features["Training_Effectiveness"] = (
        features["Training_Outcome_Numeric"] *
        features["Current_Employee_Rating"]
    )

    # --------------------------------------------------------
    # 8. TRAINING COST PER DAY
    # --------------------------------------------------------

    features["Training_Cost_Per_Day"] = (
        features["Training_Cost"] /
        features["Training_Duration(Days)"].replace(0, 1)
    )

    # --------------------------------------------------------
    # 9. PERFORMANCE-ENGAGEMENT GAP
    # --------------------------------------------------------

    features["Performance_Engagement_Gap"] = (
        features["Performance_Score_Numeric"] -
        features["Engagement_Index"]
    )

    # --------------------------------------------------------
    # 10. HIGH ENGAGEMENT FLAG
    # --------------------------------------------------------

    features["High_Engagement_Flag"] = (
        features["Engagement_Index"] >= 4
    ).astype(int)

    # --------------------------------------------------------
    # 11. LOW SATISFACTION FLAG
    # --------------------------------------------------------

    features["Low_Satisfaction_Flag"] = (
        features["Satisfaction_Score"] <= 2
    ).astype(int)

    # --------------------------------------------------------
    # 12. PERFORMANCE RISK FLAG
    # --------------------------------------------------------

    features["Performance_Risk_Flag"] = (
        features["Performance_Score"]
        .isin(["Needs Improvement", "PIP"])
    ).astype(int)

    # --------------------------------------------------------
    # 13. EMPLOYEE STATUS FLAG
    # --------------------------------------------------------

    features["Active_Employee_Flag"] = (
        features["EmployeeStatus"] == "Active"
    ).astype(int)

    # --------------------------------------------------------
    # 14. TRAINING COST CATEGORY
    # --------------------------------------------------------

    features["Training_Cost_Category"] = pd.qcut(
        features["Training_Cost"],
        q=4,
        labels=[
            "Low",
            "Medium",
            "High",
            "Very_High"
        ],
        duplicates="drop"
    )

    # --------------------------------------------------------
    # 15. ENGAGEMENT CATEGORY
    # --------------------------------------------------------

    features["Engagement_Category"] = pd.cut(
        features["Engagement_Index"],
        bins=[0, 2, 3, 4, 5],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very_High"
        ],
        include_lowest=True
    )

    # --------------------------------------------------------
    # DISPLAY NEW FEATURES
    # --------------------------------------------------------

    print("\nNew features created:")

    new_columns = [
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

    for column in new_columns:
        print(f"  [OK] {column}")

    # --------------------------------------------------------
    # CHECK MISSING VALUES
    # --------------------------------------------------------

    print("\nMissing values after feature engineering:")

    missing = features.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  [OK] No missing values")
    else:
        print(missing.to_string())

    # --------------------------------------------------------
    # CHECK INFINITE VALUES
    # --------------------------------------------------------

    print("\nInfinite values after feature engineering:")

    numeric_features = features.select_dtypes(
        include=["number"]
    )

    infinite_count = (
        numeric_features
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print(f"  Infinite values: {infinite_count}")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        FEATURES_DIR /
        "performance_engagement_features.csv"
    )

    features.to_csv(
        output_path,
        index=False
    )

    print("\nOutput:")
    print(output_path)

    print("\nFinal dataset:")
    print(f"Rows    : {features.shape[0]}")
    print(f"Columns : {features.shape[1]}")

    print("\n")
    print("=" * 80)
    print("PERFORMANCE & ENGAGEMENT FEATURE ENGINEERING COMPLETE")
    print("=" * 80)

# ============================================================
# STEP 5B.3 - O*NET SKILLS & OCCUPATION FEATURE ENGINEERING
# ============================================================

def engineer_onet_features():

    print("\n")
    print("=" * 80)
    print("STEP 5B.3 - O*NET SKILLS & OCCUPATION FEATURE ENGINEERING")
    print("=" * 80)

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    essential_path = PROCESSED_DIR / "essential_skills_clean.csv"
    software_path = PROCESSED_DIR / "software_skills_clean.csv"
    occupation_path = PROCESSED_DIR / "occupations_clean.csv"

    features_dir = PROCESSED_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. ESSENTIAL SKILLS
    # ========================================================

    print("\n")
    print("-" * 80)
    print("1. ESSENTIAL SKILLS FEATURE ENGINEERING")
    print("-" * 80)

    df = pd.read_csv(essential_path)

    print(f"\nOriginal dataset:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # Convert Scale_Name into separate Importance / Level
    # --------------------------------------------------------

    importance = df[df["Scale_Name"].str.lower() == "importance"].copy()

    level = df[df["Scale_Name"].str.lower() == "level"].copy()

    # --------------------------------------------------------
    # Rename Data_Value
    # --------------------------------------------------------

    importance = importance[
        ["O*NET-SOC_Code", "Title", "Element_ID", "Element_Name", "Data_Value"]
    ].rename(
        columns={
            "Data_Value": "Skill_Importance"
        }
    )

    level = level[
        ["O*NET-SOC_Code", "Element_ID", "Data_Value"]
    ].rename(
        columns={
            "Data_Value": "Skill_Level"
        }
    )

    # --------------------------------------------------------
    # Merge Importance + Level
    # --------------------------------------------------------

    skills = pd.merge(
        importance,
        level,
        on=["O*NET-SOC_Code", "Element_ID"],
        how="left"
    )

    # --------------------------------------------------------
    # Create combined skill strength
    # --------------------------------------------------------

    skills["Skill_Strength"] = (
        skills["Skill_Importance"] *
        skills["Skill_Level"]
    )

    # --------------------------------------------------------
    # Normalize skill strength
    # --------------------------------------------------------

    skills["Skill_Strength_Normalized"] = (
        skills["Skill_Strength"] /
        skills["Skill_Strength"].max()
    )

    # --------------------------------------------------------
    # Skill importance category
    # --------------------------------------------------------

    skills["Skill_Importance_Category"] = pd.cut(
        skills["Skill_Importance"],
        bins=[-float("inf"), 2, 4, float("inf")],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    # --------------------------------------------------------
    # Skill level category
    # --------------------------------------------------------

    skills["Skill_Level_Category"] = pd.cut(
        skills["Skill_Level"],
        bins=[-float("inf"), 2, 4, float("inf")],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    print("\nNew features created:")

    print("  [OK] Skill_Importance")
    print("  [OK] Skill_Level")
    print("  [OK] Skill_Strength")
    print("  [OK] Skill_Strength_Normalized")
    print("  [OK] Skill_Importance_Category")
    print("  [OK] Skill_Level_Category")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    essential_output = (
        features_dir /
        "essential_skills_features.csv"
    )

    skills.to_csv(
        essential_output,
        index=False
    )

    print("\nMissing values:")

    missing = skills.isnull().sum().sum()

    if missing == 0:
        print("  [OK] No missing values")
    else:
        print(f"  WARNING: {missing} missing values")

    print("\nOutput:")
    print(essential_output)

    print("\nFinal dataset:")
    print(f"Rows    : {skills.shape[0]}")
    print(f"Columns : {skills.shape[1]}")

    # ========================================================
    # 2. SOFTWARE SKILLS
    # ========================================================

    print("\n")
    print("-" * 80)
    print("2. SOFTWARE SKILLS FEATURE ENGINEERING")
    print("-" * 80)

    software = pd.read_csv(software_path)

    print(f"\nOriginal dataset:")
    print(f"Rows    : {software.shape[0]}")
    print(f"Columns : {software.shape[1]}")

    # --------------------------------------------------------
    # Binary encoding
    # --------------------------------------------------------

    software["Hot_Technology_Binary"] = (
        software["Hot_Technology"]
        .str.upper()
        .map({
            "Y": 1,
            "N": 0
        })
    )

    software["In_Demand_Binary"] = (
        software["In_Demand"]
        .str.upper()
        .map({
            "Y": 1,
            "N": 0
        })
    )

    # --------------------------------------------------------
    # Technology demand score
    # --------------------------------------------------------

    software["Technology_Demand_Score"] = (
        software["Hot_Technology_Binary"] +
        software["In_Demand_Binary"]
    )

    # --------------------------------------------------------
    # Technology category
    # --------------------------------------------------------

    software["Technology_Category"] = software[
        "Technology_Demand_Score"
    ].map({
        0: "Standard",
        1: "Relevant",
        2: "High Demand"
    })

    # --------------------------------------------------------
    # Software name cleaning
    # --------------------------------------------------------

    software["Software_Name"] = (
        software["Workplace_Example"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Software category cleaning
    # --------------------------------------------------------

    software["Software_Category"] = (
        software["Element_Name"]
        .astype(str)
        .str.strip()
    )

    print("\nNew features created:")

    print("  [OK] Hot_Technology_Binary")
    print("  [OK] In_Demand_Binary")
    print("  [OK] Technology_Demand_Score")
    print("  [OK] Technology_Category")
    print("  [OK] Software_Name")
    print("  [OK] Software_Category")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    software_output = (
        features_dir /
        "software_skills_features.csv"
    )

    software.to_csv(
        software_output,
        index=False
    )

    print("\nMissing values:")

    missing = software.isnull().sum().sum()

    if missing == 0:
        print("  [OK] No missing values")
    else:
        print(f"  WARNING: {missing} missing values")

    print("\nOutput:")
    print(software_output)

    print("\nFinal dataset:")
    print(f"Rows    : {software.shape[0]}")
    print(f"Columns : {software.shape[1]}")

    # ========================================================
    # 3. OCCUPATIONS
    # ========================================================

    print("\n")
    print("-" * 80)
    print("3. OCCUPATION FEATURE ENGINEERING")
    print("-" * 80)

    occupations = pd.read_csv(occupation_path)

    print(f"\nOriginal dataset:")
    print(f"Rows    : {occupations.shape[0]}")
    print(f"Columns : {occupations.shape[1]}")

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    occupations["Occupation_Title"] = (
        occupations["Title"]
        .astype(str)
        .str.strip()
    )

    occupations["Occupation_Description"] = (
        occupations["Description"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Description length
    # --------------------------------------------------------

    occupations["Description_Length"] = (
        occupations["Occupation_Description"]
        .str.len()
    )

    # --------------------------------------------------------
    # Word count
    # --------------------------------------------------------

    occupations["Description_Word_Count"] = (
        occupations["Occupation_Description"]
        .str.split()
        .str.len()
    )

    # --------------------------------------------------------
    # Management indicator
    # --------------------------------------------------------

    management_keywords = [
        "manager",
        "management",
        "director",
        "executive",
        "supervisor"
    ]

    occupations["Management_Role_Flag"] = (
        occupations["Occupation_Title"]
        .str.lower()
        .apply(
            lambda x:
            int(
                any(
                    keyword in x
                    for keyword in management_keywords
                )
            )
        )
    )

    # --------------------------------------------------------
    # Technology indicator
    # --------------------------------------------------------

    technology_keywords = [
        "computer",
        "software",
        "information",
        "data",
        "database",
        "network",
        "developer",
        "programmer",
        "web",
        "technology",
        "systems",
        "cybersecurity"
    ]

    occupations["Technology_Role_Flag"] = (
        occupations["Occupation_Title"]
        .str.lower()
        .apply(
            lambda x:
            int(
                any(
                    keyword in x
                    for keyword in technology_keywords
                )
            )
        )
    )

    print("\nNew features created:")

    print("  [OK] Occupation_Title")
    print("  [OK] Occupation_Description")
    print("  [OK] Description_Length")
    print("  [OK] Description_Word_Count")
    print("  [OK] Management_Role_Flag")
    print("  [OK] Technology_Role_Flag")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    occupation_output = (
        features_dir /
        "occupations_features.csv"
    )

    occupations.to_csv(
        occupation_output,
        index=False
    )

    print("\nMissing values:")

    missing = occupations.isnull().sum().sum()

    if missing == 0:
        print("  [OK] No missing values")
    else:
        print(f"  WARNING: {missing} missing values")

    print("\nOutput:")
    print(occupation_output)

    print("\nFinal dataset:")
    print(f"Rows    : {occupations.shape[0]}")
    print(f"Columns : {occupations.shape[1]}")

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("STEP 5B.3 O*NET FEATURE ENGINEERING COMPLETE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    engineer_attrition_features()
    engineer_performance_features()
    engineer_onet_features()




if __name__ == "__main__":
    main()