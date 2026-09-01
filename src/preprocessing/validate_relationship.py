import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    attrition = pd.read_csv(
        DATA_DIR / "attrition_clean.csv"
    )

    performance = pd.read_csv(
        DATA_DIR / "performance_engagement_clean.csv"
    )

    occupations = pd.read_csv(
        DATA_DIR / "occupations_clean.csv"
    )

    essential = pd.read_csv(
        DATA_DIR / "essential_skills_clean.csv"
    )

    software = pd.read_csv(
        DATA_DIR / "software_skills_clean.csv"
    )

    return (
        attrition,
        performance,
        occupations,
        essential,
        software
    )


# ============================================================
# PRINT SECTION
# ============================================================

def section(title):

    print("\n")
    print("=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# OCCUPATION RELATIONSHIPS
# ============================================================

def validate_occupation_relationships(
    occupations,
    essential,
    software
):

    section("OCCUPATION RELATIONSHIPS")

    occupation_codes = set(
        occupations["O*NET-SOC_Code"]
    )

    essential_codes = set(
        essential["O*NET-SOC_Code"]
    )

    software_codes = set(
        software["O*NET-SOC_Code"]
    )

    print(
        f"Occupation codes       : "
        f"{len(occupation_codes)}"
    )

    print(
        f"Essential skill codes  : "
        f"{len(essential_codes)}"
    )

    print(
        f"Software skill codes   : "
        f"{len(software_codes)}"
    )

    # --------------------------------------------------------
    # Essential → Occupation
    # --------------------------------------------------------

    essential_match = (
        essential_codes
        .intersection(occupation_codes)
    )

    print(
        f"\nEssential codes matching occupations: "
        f"{len(essential_match)}"
    )

    print(
        f"Essential codes NOT matching: "
        f"{len(essential_codes - occupation_codes)}"
    )

    # --------------------------------------------------------
    # Software → Occupation
    # --------------------------------------------------------

    software_match = (
        software_codes
        .intersection(occupation_codes)
    )

    print(
        f"\nSoftware codes matching occupations: "
        f"{len(software_match)}"
    )

    print(
        f"Software codes NOT matching: "
        f"{len(software_codes - occupation_codes)}"
    )


# ============================================================
# TITLE CONSISTENCY
# ============================================================

def validate_title_consistency(
    occupations,
    essential,
    software
):

    section("TITLE CONSISTENCY")

    occupation_titles = set(
        occupations["Title"]
        .str.strip()
        .str.lower()
    )

    essential_titles = set(
        essential["Title"]
        .str.strip()
        .str.lower()
    )

    software_titles = set(
        software["Title"]
        .str.strip()
        .str.lower()
    )

    print(
        f"Occupation titles : "
        f"{len(occupation_titles)}"
    )

    print(
        f"Essential titles  : "
        f"{len(essential_titles)}"
    )

    print(
        f"Software titles   : "
        f"{len(software_titles)}"
    )

    print(
        "\nEssential titles matching occupation titles:",
        len(
            essential_titles
            .intersection(occupation_titles)
        )
    )

    print(
        "Software titles matching occupation titles:",
        len(
            software_titles
            .intersection(occupation_titles)
        )
    )


# ============================================================
# SKILL STRUCTURE
# ============================================================

def validate_skill_structure(
    essential,
    software
):

    section("SKILL STRUCTURE")

    print(
        "Essential skills:"
    )

    print(
        essential[
            [
                "Element_ID",
                "Element_Name",
                "Scale_Name"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["Element_ID", "Scale_Name"]
        )
        .to_string(index=False)
    )

    print(
        "\nSoftware skill categories:"
    )

    print(
        software[
            [
                "Element_ID",
                "Element_Name"
            ]
        ]
        .drop_duplicates()
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# EMPLOYEE DATA RELATIONSHIP
# ============================================================

def validate_employee_data(
    attrition,
    performance
):

    section("EMPLOYEE DATA RELATIONSHIP")

    print(
        f"Attrition employees    : "
        f"{len(attrition)}"
    )

    print(
        f"Performance employees  : "
        f"{len(performance)}"
    )

    # --------------------------------------------------------
    # Attrition has no explicit employee ID
    # --------------------------------------------------------

    print(
        "\nAttrition dataset has no Employee_ID column."
    )

    print(
        "Therefore we will NOT force a row-level merge "
        "between attrition and performance data."
    )

    # --------------------------------------------------------
    # Performance IDs
    # --------------------------------------------------------

    if "Employee_ID" in performance.columns:

        unique_ids = performance[
            "Employee_ID"
        ].nunique()

        print(
            f"\nPerformance Employee_ID count : "
            f"{unique_ids}"
        )

        print(
            f"Duplicate Employee_ID rows : "
            f"{len(performance) - unique_ids}"
        )


# ============================================================
# SOFTWARE DEMAND ANALYSIS
# ============================================================

def validate_software_demand(software):

    section("SOFTWARE DEMAND ANALYSIS")

    demand = (
        software
        .groupby("Workplace_Example")
        .agg(
            records=("Workplace_Example", "size"),
            occupations=("O*NET-SOC_Code", "nunique"),
            hot_count=(
                "Hot_Technology",
                lambda x: (x == "Y").sum()
            ),
            demand_count=(
                "In_Demand",
                lambda x: (x == "Y").sum()
            )
        )
        .sort_values(
            "occupations",
            ascending=False
        )
        .head(20)
    )

    print(
        demand.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "AI WORKFORCE INTELLIGENCE PLATFORM\n"
        "STEP 5B — DATASET RELATIONSHIP VALIDATION"
    )

    (
        attrition,
        performance,
        occupations,
        essential,
        software
    ) = load_data()

    validate_occupation_relationships(
        occupations,
        essential,
        software
    )

    validate_title_consistency(
        occupations,
        essential,
        software
    )

    validate_skill_structure(
        essential,
        software
    )

    validate_employee_data(
        attrition,
        performance
    )

    validate_software_demand(
        software
    )

    section(
        "STEP 5B VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()