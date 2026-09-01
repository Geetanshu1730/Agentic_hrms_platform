import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "attrition_clean.csv"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "attrition"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

print("=" * 70)
print("AI WORKFORCE INTELLIGENCE PLATFORM")
print("ATTRITION ML DATA PREPARATION")
print("=" * 70)

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Attrition"]
)

y = df["Attrition"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("\nAfter encoding:")
print("Features:", X.shape)


# ============================================================
# SCALE NUMERICAL FEATURES
# ============================================================

scaler = StandardScaler()

X[numerical_columns] = scaler.fit_transform(
    X[numerical_columns]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain/Test split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# SAVE TRAIN / TEST DATA
# ============================================================

X_train.to_csv(
    MODEL_DIR / "X_train.csv",
    index=False
)

X_test.to_csv(
    MODEL_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    MODEL_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    MODEL_DIR / "y_test.csv",
    index=False
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)


# ============================================================
# SAVE FEATURE NAMES
# ============================================================

feature_names = X.columns.tolist()

joblib.dump(
    feature_names,
    MODEL_DIR / "feature_names.pkl"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SAVED FILES")
print("=" * 70)

print("\nX_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")
print("scaler.pkl")
print("feature_names.pkl")

print("\nLocation:")
print(MODEL_DIR)

print("\nML data preparation complete!")
print("=" * 70)