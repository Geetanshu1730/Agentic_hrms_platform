import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "attrition"

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "attrition_clean.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "attrition_model.pt"
)

OUTPUT_FILE = (
    MODEL_DIR
    / "attrition_predictions.csv"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class AttritionNetwork(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 128),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD ML-READY DATA
# ============================================================

print("=" * 70)
print("AI WORKFORCE INTELLIGENCE PLATFORM")
print("ATTRITION RISK PREDICTION")
print("=" * 70)

print("\nLoading test data...")

X_test = pd.read_csv(
    MODEL_DIR / "X_test.csv"
)

print("X_test:", X_test.shape)


# ============================================================
# LOAD ORIGINAL EMPLOYEE DATA
# ============================================================

original_df = pd.read_csv(
    DATA_FILE
)

# The test rows were created using random_state=42.
# Recreate the same train/test indices.

from sklearn.model_selection import train_test_split

indices = original_df.index

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=original_df["Attrition"]
)

employee_test = original_df.loc[
    test_indices
].copy()

employee_test = employee_test.reset_index(
    drop=True
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

input_size = checkpoint["input_size"]

model = AttritionNetwork(
    input_size
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# CONVERT DATA TO TENSOR
# ============================================================

X_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
).to(device)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

with torch.no_grad():

    logits = model(
        X_tensor
    )

    probabilities = torch.sigmoid(
        logits
    )

probabilities = (
    probabilities
    .cpu()
    .numpy()
    .flatten()
)


# ============================================================
# CREATE RISK LEVELS
# ============================================================

def get_risk_level(probability):

    if probability < 0.40:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


risk_levels = [
    get_risk_level(p)
    for p in probabilities
]


# ============================================================
# BUILD OUTPUT
# ============================================================

employee_test["Attrition_Probability"] = (
    probabilities
)

employee_test["Attrition_Risk"] = (
    risk_levels
)

employee_test["Attrition_Probability"] = (
    employee_test["Attrition_Probability"] * 100
).round(2)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

employee_test.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ATTRITION RISK SUMMARY")
print("=" * 70)

print(
    "\nLOW:",
    sum(
        employee_test["Attrition_Risk"] == "LOW"
    )
)

print(
    "MEDIUM:",
    sum(
        employee_test["Attrition_Risk"] == "MEDIUM"
    )
)

print(
    "HIGH:",
    sum(
        employee_test["Attrition_Risk"] == "HIGH"
    )
)

print("\nAverage predicted risk:",
      round(
          employee_test[
              "Attrition_Probability"
          ].mean(),
          2
      ),
      "%"
)

print("\nPredictions saved to:")

print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("ATTRITION RISK PREDICTION COMPLETE")
print("=" * 70)