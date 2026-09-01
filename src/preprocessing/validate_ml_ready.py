
import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# AI WORKFORCE INTELLIGENCE PLATFORM
# STEP 5E - ML-READY DATA VALIDATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

ML_READY_DIR = os.path.join(
    BASE_DIR, "data", "processed", "ml_ready"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models", "preprocessing"
)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def validate_dataset(
    name,
    x_train_file,
    x_test_file,
    y_train_file,
    y_test_file,
    preprocessor_file
):

    print_header(f"VALIDATING: {name}")

    # ========================================================
    # 1. FILE EXISTENCE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("1. FILE EXISTENCE CHECK")
    print("-" * 80)

    files = {
        "X_train": os.path.join(ML_READY_DIR, x_train_file),
        "X_test": os.path.join(ML_READY_DIR, x_test_file),
        "y_train": os.path.join(ML_READY_DIR, y_train_file),
        "y_test": os.path.join(ML_READY_DIR, y_test_file),
        "Preprocessor": os.path.join(MODEL_DIR, preprocessor_file)
    }

    all_files_exist = True

    for label, path in files.items():

        if os.path.exists(path):
            print(f"[OK] {label}: {path}")
        else:
            print(f"[ERROR] Missing {label}: {path}")
            all_files_exist = False

    if not all_files_exist:
        print("\n[ERROR] Required files are missing.")
        return False

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print("\n" + "-" * 80)
    print("2. LOADING ML-READY DATA")
    print("-" * 80)

    try:

        X_train = pd.read_csv(files["X_train"])
        X_test = pd.read_csv(files["X_test"])
        y_train = pd.read_csv(files["y_train"])
        y_test = pd.read_csv(files["y_test"])

        print("[OK] All ML-ready datasets loaded successfully.")

    except Exception as e:

        print(f"[ERROR] Could not load datasets: {e}")
        return False

    # ========================================================
    # 3. DATASET SHAPES
    # ========================================================

    print("\n" + "-" * 80)
    print("3. DATASET SHAPES")
    print("-" * 80)

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    # ========================================================
    # 4. FEATURE COUNT CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("4. FEATURE COUNT CHECK")
    print("-" * 80)

    if X_train.shape[1] == X_test.shape[1]:

        print(
            f"[OK] Train/Test feature count matches: "
            f"{X_train.shape[1]}"
        )

    else:

        print("[ERROR] Train/Test feature count mismatch.")
        return False

    # ========================================================
    # 5. FEATURE COLUMN MATCH CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("5. FEATURE COLUMN CHECK")
    print("-" * 80)

    if list(X_train.columns) == list(X_test.columns):

        print(
            "[OK] Train and test feature columns "
            "match exactly."
        )

    else:

        print(
            "[ERROR] Train and test feature columns "
            "do not match."
        )

        return False

    # ========================================================
    # 6. TARGET SIZE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("6. TARGET SIZE CHECK")
    print("-" * 80)

    if len(X_train) == len(y_train):

        print(
            "[OK] X_train and y_train row counts match."
        )

    else:

        print(
            "[ERROR] X_train / y_train row count mismatch."
        )

        return False

    if len(X_test) == len(y_test):

        print(
            "[OK] X_test and y_test row counts match."
        )

    else:

        print(
            "[ERROR] X_test / y_test row count mismatch."
        )

        return False

    # ========================================================
    # 7. MISSING VALUE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("7. MISSING VALUE CHECK")
    print("-" * 80)

    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()

    train_target_missing = y_train.isna().sum().sum()
    test_target_missing = y_test.isna().sum().sum()

    print(
        f"X_train missing values : "
        f"{train_missing}"
    )

    print(
        f"X_test missing values  : "
        f"{test_missing}"
    )

    print(
        f"y_train missing values : "
        f"{train_target_missing}"
    )

    print(
        f"y_test missing values  : "
        f"{test_target_missing}"
    )

    if (
        train_missing == 0
        and test_missing == 0
        and train_target_missing == 0
        and test_target_missing == 0
    ):

        print("[OK] No missing values found.")

    else:

        print("[ERROR] Missing values detected.")
        return False

    # ========================================================
    # 8. INFINITE VALUE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("8. INFINITE VALUE CHECK")
    print("-" * 80)

    try:

        train_numeric = X_train.select_dtypes(
            include=np.number
        )

        test_numeric = X_test.select_dtypes(
            include=np.number
        )

        train_inf = np.isinf(
            train_numeric.to_numpy()
        ).sum()

        test_inf = np.isinf(
            test_numeric.to_numpy()
        ).sum()

        print(
            f"X_train infinite values : "
            f"{train_inf}"
        )

        print(
            f"X_test infinite values  : "
            f"{test_inf}"
        )

        if train_inf == 0 and test_inf == 0:

            print("[OK] No infinite values found.")

        else:

            print("[ERROR] Infinite values detected.")
            return False

    except Exception as e:

        print(
            f"[ERROR] Infinite value check failed: {e}"
        )

        return False

    # ========================================================
    # 9. DUPLICATE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("9. DUPLICATE CHECK")
    print("-" * 80)

    train_duplicates = X_train.duplicated().sum()
    test_duplicates = X_test.duplicated().sum()

    print(
        f"X_train duplicate rows : "
        f"{train_duplicates}"
    )

    print(
        f"X_test duplicate rows  : "
        f"{test_duplicates}"
    )

    if train_duplicates == 0 and test_duplicates == 0:

        print("[OK] No duplicate feature rows.")

    else:

        print(
            "[WARNING] Duplicate feature rows detected."
        )

    # ========================================================
    # 10. NUMERIC FEATURE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("10. NUMERIC FEATURE CHECK")
    print("-" * 80)

    train_non_numeric = X_train.select_dtypes(
        exclude=np.number
    ).shape[1]

    test_non_numeric = X_test.select_dtypes(
        exclude=np.number
    ).shape[1]

    print(
        f"Non-numeric columns in X_train : "
        f"{train_non_numeric}"
    )

    print(
        f"Non-numeric columns in X_test  : "
        f"{test_non_numeric}"
    )

    if train_non_numeric == 0 and test_non_numeric == 0:

        print(
            "[OK] All ML-ready features are numeric."
        )

    else:

        print(
            "[ERROR] Non-numeric features detected."
        )

        return False

    # ========================================================
    # 11. TARGET DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 80)
    print("11. TARGET DISTRIBUTION")
    print("-" * 80)

    target_column = y_train.columns[0]

    print("\nTraining target:")

    print(
        y_train[target_column].value_counts()
    )

    print("\nTraining percentage:")

    print(
        (
            y_train[target_column]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    print("\nTest target:")

    print(
        y_test[target_column].value_counts()
    )

    print("\nTest percentage:")

    print(
        (
            y_test[target_column]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    # ========================================================
    # 12. TARGET VALUE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("12. TARGET VALUE CHECK")
    print("-" * 80)

    train_targets = set(
        y_train[target_column].unique()
    )

    test_targets = set(
        y_test[target_column].unique()
    )

    print(
        f"Train target values: "
        f"{sorted(train_targets)}"
    )

    print(
        f"Test target values : "
        f"{sorted(test_targets)}"
    )

    if train_targets == test_targets:

        print(
            "[OK] Train/Test contain the same "
            "target classes."
        )

    else:

        print(
            "[WARNING] Train/Test target classes differ."
        )

    # ========================================================
    # 13. TARGET DATA TYPE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("13. TARGET DATA TYPE CHECK")
    print("-" * 80)

    print(
        f"Target dtype: "
        f"{y_train[target_column].dtype}"
    )

    if pd.api.types.is_numeric_dtype(
        y_train[target_column]
    ):

        print("[OK] Target is numeric.")

    else:

        print(
            "[WARNING] Target is not numeric."
        )

    # ========================================================
    # 14. PREPROCESSOR CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("14. PREPROCESSOR CHECK")
    print("-" * 80)

    try:

        preprocessor = joblib.load(
            files["Preprocessor"]
        )

        print(
            "[OK] Preprocessing pipeline loaded "
            "successfully."
        )

        print(
            f"Type: "
            f"{type(preprocessor).__name__}"
        )

    except Exception as e:

        print(
            f"[ERROR] Could not load preprocessor: {e}"
        )

        return False

    # ========================================================
    # 15. PREPROCESSOR FEATURE COUNT CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("15. PREPROCESSOR FEATURE COUNT CHECK")
    print("-" * 80)

    try:

        preprocessor_feature_count = len(
            preprocessor.get_feature_names_out()
        )

        ml_ready_feature_count = X_train.shape[1]

        print(
            f"Preprocessor output features : "
            f"{preprocessor_feature_count}"
        )

        print(
            f"ML-ready CSV features        : "
            f"{ml_ready_feature_count}"
        )

        if (
            preprocessor_feature_count
            == ml_ready_feature_count
        ):

            print(
                "[OK] Preprocessor feature count "
                "matches ML-ready data."
            )

        else:

            print(
                "[WARNING] Preprocessor feature count "
                "does not match ML-ready CSV."
            )

            print(
                "This can happen if the CSV contains "
                "column names generated during transformation."
            )

    except Exception as e:

        print(
            f"[WARNING] Could not inspect "
            f"preprocessor features: {e}"
        )

    # ========================================================
    # 16. ML-READY NaN CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("16. FINAL NaN CHECK")
    print("-" * 80)

    try:

        train_nan = np.isnan(
            X_train.to_numpy(dtype=float)
        ).sum()

        test_nan = np.isnan(
            X_test.to_numpy(dtype=float)
        ).sum()

        print(
            f"X_train NaN values : "
            f"{train_nan}"
        )

        print(
            f"X_test NaN values  : "
            f"{test_nan}"
        )

        if train_nan == 0 and test_nan == 0:

            print(
                "[OK] ML-ready data contains no NaN values."
            )

        else:

            print(
                "[ERROR] NaN values detected."
            )

            return False

    except Exception as e:

        print(
            f"[ERROR] NaN check failed: {e}"
        )

        return False

    # ========================================================
    # 17. FINAL INFINITE CHECK
    # ========================================================

    print("\n" + "-" * 80)
    print("17. FINAL INFINITE CHECK")
    print("-" * 80)

    try:

        train_inf = np.isinf(
            X_train.to_numpy(dtype=float)
        ).sum()

        test_inf = np.isinf(
            X_test.to_numpy(dtype=float)
        ).sum()

        print(
            f"X_train infinite values : "
            f"{train_inf}"
        )

        print(
            f"X_test infinite values  : "
            f"{test_inf}"
        )

        if train_inf == 0 and test_inf == 0:

            print(
                "[OK] ML-ready data contains no "
                "infinite values."
            )

        else:

            print(
                "[ERROR] Infinite values detected."
            )

            return False

    except Exception as e:

        print(
            f"[ERROR] Infinite check failed: {e}"
        )

        return False

    # ========================================================
    # 18. FINAL VALIDATION RESULT
    # ========================================================

    print("\n" + "=" * 80)

    print(
        f"[OK] {name.upper()} "
        f"ML-READY VALIDATION PASSED"
    )

    print("=" * 80)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "AI WORKFORCE INTELLIGENCE PLATFORM\n"
        "STEP 5E - ML-READY DATA VALIDATION"
    )

    # ========================================================
    # ATTRITION
    # ========================================================

    attrition_ok = validate_dataset(

        name="ATTRITION",

        x_train_file="attrition_X_train.csv",

        x_test_file="attrition_X_test.csv",

        y_train_file="attrition_y_train.csv",

        y_test_file="attrition_y_test.csv",

        preprocessor_file="attrition_preprocessor.joblib"
    )

    # ========================================================
    # PERFORMANCE & ENGAGEMENT
    # ========================================================

    performance_ok = validate_dataset(

        name="PERFORMANCE & ENGAGEMENT",

        x_train_file="performance_X_train.csv",

        x_test_file="performance_X_test.csv",

        y_train_file="performance_y_train.csv",

        y_test_file="performance_y_test.csv",

        preprocessor_file="performance_preprocessor.joblib"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")

    print("=" * 80)

    print("STEP 5E VALIDATION SUMMARY")

    print("=" * 80)

    print(
        f"Attrition ML data              : "
        f"{'[OK] PASSED' if attrition_ok else '[ERROR] FAILED'}"
    )

    print(
        f"Performance ML data            : "
        f"{'[OK] PASSED' if performance_ok else '[ERROR] FAILED'}"
    )

    if attrition_ok and performance_ok:

        print("\n" + "=" * 80)

        print(
            "[OK] ALL ML-READY DATASETS "
            "PASSED VALIDATION"
        )

        print("=" * 80)

        print("\nData preprocessing pipeline is COMPLETE.")

        print(
            "Next step: STEP 6 - MODEL DEVELOPMENT"
        )

    else:

        print("\n" + "=" * 80)

        print(
            "[ERROR] VALIDATION FAILED"
        )

        print(
            "Fix the errors above before starting "
            "model development."
        )

        print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
