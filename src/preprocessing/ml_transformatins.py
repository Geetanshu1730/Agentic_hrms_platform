"""
================================================================================
AI WORKFORCE INTELLIGENCE PLATFORM
STEP 5D - ML-READY TRANSFORMATIONS
================================================================================

Purpose:
    Convert engineered datasets into ML-ready train/test datasets.

Tasks:
    1. Separate X and y
    2. Remove identifiers/personal information
    3. Encode categorical features
    4. Scale numerical features
    5. Split into train/test sets
    6. Handle Attrition class imbalance using class weights
    7. Save preprocessing pipelines
    8. Save ML-ready datasets
================================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ==============================================================================
# PATHS
# ==============================================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features"
)

ML_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_ready"
)

PIPELINE_DIR = os.path.join(
    BASE_DIR,
    "models",
    "preprocessing"
)

os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(PIPELINE_DIR, exist_ok=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_header(title):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def save_csv(df, filename):
    path = os.path.join(ML_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[OK] Saved: {path}")
    return path


# ==============================================================================
# STEP 5D.1 - ATTRITION ML TRANSFORMATION
# ==============================================================================

def transform_attrition():

    print_header("STEP 5D.1 - ATTRITION ML-READY TRANSFORMATION")

    input_path = os.path.join(
        FEATURE_DIR,
        "attrition_features.csv"
    )

    df = pd.read_csv(input_path)

    print("\nOriginal dataset:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------------------------
    # TARGET
    # --------------------------------------------------------------------------

    target = "Attrition"

    X = df.drop(columns=[target])
    y = df[target]

    print("\nTarget:")
    print(f"  {target}")

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------------------------------
    # REMOVE COLUMNS THAT SHOULD NOT BE USED FOR ML
    # --------------------------------------------------------------------------

    columns_to_remove = []

    # These are raw categorical columns that are represented by engineered
    # versions or are not necessary for the initial attrition model.
    #
    # We keep useful HR features such as Department, JobRole, OverTime, etc.

    # --------------------------------------------------------------------------
    # IDENTIFY DATA TYPES
    # --------------------------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print("\nCategorical features:")
    for column in categorical_columns:
        print(f"  [CAT] {column}")

    print("\nNumerical features:")
    for column in numerical_columns:
        print(f"  [NUM] {column}")

    # --------------------------------------------------------------------------
    # PREPROCESSING PIPELINE
    # --------------------------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    # --------------------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test : {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test : {y_test.shape}")

    # --------------------------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------------------------

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # --------------------------------------------------------------------------
    # GET FEATURE NAMES
    # --------------------------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names
    )

    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names
    )

    # --------------------------------------------------------------------------
    # SAVE TARGETS
    # --------------------------------------------------------------------------

    y_train = pd.Series(
        y_train.values,
        name="Attrition"
    )

    y_test = pd.Series(
        y_test.values,
        name="Attrition"
    )

    # --------------------------------------------------------------------------
    # SAVE DATASETS
    # --------------------------------------------------------------------------

    save_csv(
        X_train_transformed,
        "attrition_X_train.csv"
    )

    save_csv(
        X_test_transformed,
        "attrition_X_test.csv"
    )

    save_csv(
        y_train,
        "attrition_y_train.csv"
    )

    save_csv(
        y_test,
        "attrition_y_test.csv"
    )

    # --------------------------------------------------------------------------
    # SAVE PIPELINE
    # --------------------------------------------------------------------------

    pipeline_path = os.path.join(
        PIPELINE_DIR,
        "attrition_preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        pipeline_path
    )

    print(f"[OK] Saved preprocessing pipeline:")
    print(f"     {pipeline_path}")

    # --------------------------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------------------------

    class_counts = y_train.value_counts()

    total = len(y_train)

    class_weights = {}

    for class_value, count in class_counts.items():
        class_weights[int(class_value)] = total / (
            len(class_counts) * count
        )

    class_weights_path = os.path.join(
        PIPELINE_DIR,
        "attrition_class_weights.joblib"
    )

    joblib.dump(
        class_weights,
        class_weights_path
    )

    print("\nClass weights:")
    for key, value in class_weights.items():
        print(f"  Class {key}: {value:.4f}")

    print(f"\n[OK] Saved class weights:")
    print(f"     {class_weights_path}")

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------

    print("\nTransformed feature count:")
    print(f"  {X_train_transformed.shape[1]}")

    print("\nMissing values:")
    print(f"  Train: {X_train_transformed.isna().sum().sum()}")
    print(f"  Test : {X_test_transformed.isna().sum().sum()}")

    print("\nInfinite values:")
    print(
        f"  Train: {np.isinf(X_train_transformed.values).sum()}"
    )
    print(
        f"  Test : {np.isinf(X_test_transformed.values).sum()}"
    )

    print("\n[OK] ATTRITION ML TRANSFORMATION COMPLETE")


# ==============================================================================
# STEP 5D.2 - PERFORMANCE & ENGAGEMENT ML TRANSFORMATION
# ==============================================================================

def transform_performance():

    print_header(
        "STEP 5D.2 - PERFORMANCE & ENGAGEMENT ML-READY TRANSFORMATION"
    )

    input_path = os.path.join(
        FEATURE_DIR,
        "performance_engagement_features.csv"
    )

    df = pd.read_csv(input_path)

    print("\nOriginal dataset:")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------------------------
    # REMOVE PERSONAL / IDENTIFIER COLUMNS
    # --------------------------------------------------------------------------

    columns_to_remove = [
        "Unnamed:_0",
        "FirstName",
        "LastName",
        "ADEmail",
        "Employee_ID",
        "DOB",
        "Supervisor",
        "TerminationDescription",
        "Location",
        "Trainer"
    ]

    columns_to_remove = [
        col for col in columns_to_remove
        if col in df.columns
    ]

    print("\nRemoving identifier/personal columns:")

    for column in columns_to_remove:
        print(f"  [REMOVE] {column}")

    df = df.drop(
        columns=columns_to_remove
    )

    # --------------------------------------------------------------------------
    # REMOVE RAW DATE COLUMNS
    # --------------------------------------------------------------------------

    date_columns = [
        "StartDate",
        "ExitDate",
        "Survey_Date",
        "Training_Date"
    ]

    date_columns = [
        col for col in date_columns
        if col in df.columns
    ]

    print("\nRemoving raw date columns:")

    for column in date_columns:
        print(f"  [REMOVE] {column}")

    df = df.drop(
        columns=date_columns
    )

    # --------------------------------------------------------------------------
    # TARGET
    #
    # For the first ML task we predict Performance_Score.
    # --------------------------------------------------------------------------

    target = "Performance_Score"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # --------------------------------------------------------------------------
    # CONVERT PERFORMANCE TARGET TO NUMERIC
    # --------------------------------------------------------------------------

    performance_mapping = {
        "Needs Improvement": 1,
        "PIP": 2,
        "Fully Meets": 3,
        "Exceeds": 4
    }

    y = y.map(performance_mapping)

    print("\nTarget:")
    print("  Performance_Score")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------------------------------
    # REMOVE OTHER TARGET-LEAKAGE / DERIVED FEATURES
    #
    # Performance_Risk_Flag is directly based on performance.
    # Performance_Engagement_Gap also contains performance information.
    # Performance_Score_Numeric is the numeric form of the target.
    # --------------------------------------------------------------------------

    leakage_columns = [
        "Performance_Score_Numeric",
        "Performance_Risk_Flag",
        "Performance_Engagement_Gap"
    ]

    leakage_columns = [
        col for col in leakage_columns
        if col in X.columns
    ]

    print("\nRemoving target-leakage columns:")

    for column in leakage_columns:
        print(f"  [REMOVE] {column}")

    X = X.drop(
        columns=leakage_columns
    )

    # --------------------------------------------------------------------------
    # IDENTIFY DATA TYPES
    # --------------------------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    print("\nCategorical features:")
    for column in categorical_columns:
        print(f"  [CAT] {column}")

    print("\nNumerical features:")
    for column in numerical_columns:
        print(f"  [NUM] {column}")

    # --------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    # --------------------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test : {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test : {y_test.shape}")

    # --------------------------------------------------------------------------
    # FIT PREPROCESSOR ONLY ON TRAINING DATA
    # --------------------------------------------------------------------------

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    X_train_transformed = pd.DataFrame(
        X_train_transformed,
        columns=feature_names
    )

    X_test_transformed = pd.DataFrame(
        X_test_transformed,
        columns=feature_names
    )

    # --------------------------------------------------------------------------
    # SAVE TARGETS
    # --------------------------------------------------------------------------

    y_train = pd.Series(
        y_train.values,
        name="Performance_Score"
    )

    y_test = pd.Series(
        y_test.values,
        name="Performance_Score"
    )

    # --------------------------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------------------------

    save_csv(
        X_train_transformed,
        "performance_X_train.csv"
    )

    save_csv(
        X_test_transformed,
        "performance_X_test.csv"
    )

    save_csv(
        y_train,
        "performance_y_train.csv"
    )

    save_csv(
        y_test,
        "performance_y_test.csv"
    )

    # --------------------------------------------------------------------------
    # SAVE PREPROCESSOR
    # --------------------------------------------------------------------------

    pipeline_path = os.path.join(
        PIPELINE_DIR,
        "performance_preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        pipeline_path
    )

    print(f"[OK] Saved preprocessing pipeline:")
    print(f"     {pipeline_path}")

    # --------------------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------------------

    print("\nTransformed feature count:")
    print(f"  {X_train_transformed.shape[1]}")

    print("\nMissing values:")
    print(f"  Train: {X_train_transformed.isna().sum().sum()}")
    print(f"  Test : {X_test_transformed.isna().sum().sum()}")

    print("\nInfinite values:")
    print(
        f"  Train: {np.isinf(X_train_transformed.values).sum()}"
    )
    print(
        f"  Test : {np.isinf(X_test_transformed.values).sum()}"
    )

    print("\n[OK] PERFORMANCE ML TRANSFORMATION COMPLETE")


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print_header(
        "STEP 5D - ML-READY TRANSFORMATIONS"
    )

    transform_attrition()

    transform_performance()

    print_header(
        "STEP 5D COMPLETE"
    )

    print("\n[OK] Attrition data transformed")
    print("[OK] Performance data transformed")
    print("[OK] Train/test datasets created")
    print("[OK] Preprocessing pipelines saved")
    print("[OK] Class weights saved for attrition")
    print("\nNext step: STEP 5E - ML-READY DATA VALIDATION")


if __name__ == "__main__":
    main()