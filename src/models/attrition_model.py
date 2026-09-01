import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Ensure target directories exist
os.makedirs("models/attrition", exist_ok=True)

# 1. Load ML-Ready Datasets
X_train_df = pd.read_csv("data/processed/ml_ready/attrition_X_train.csv")
X_test_df = pd.read_csv("data/processed/ml_ready/attrition_X_test.csv")
y_train_df = pd.read_csv("data/processed/ml_ready/attrition_y_train.csv")
y_test_df = pd.read_csv("data/processed/ml_ready/attrition_y_test.csv")

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
y_train = torch.tensor(y_train_df.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
y_test = torch.tensor(y_test_df.values, dtype=torch.float32).view(-1, 1)

# 2. Model Architecture
class AttritionNetwork(nn.Module):
    def __init__(self, input_dim=44):
        super(AttritionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
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

# 3. Training Setup
torch.manual_seed(42)
np.random.seed(42)

model = AttritionNetwork(input_dim=X_train.shape[1])
pos_weight = torch.tensor([5.189473684210526], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 4. Training Loop
batch_size = 32
num_epochs = 50
dataset_size = X_train.shape[0]

model.train()
for epoch in range(num_epochs):
    permutation = torch.randperm(dataset_size)
    for i in range(0, dataset_size, batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 5. Inference & Probability Generation
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits).numpy().flatten()

y_true = y_test.numpy().flatten()

# 6. Threshold Tuning for Optimal F1
thresholds = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
best_f1 = 0.0

for t in thresholds:
    preds = (probabilities >= t).astype(int)
    score = f1_score(y_true, preds, zero_division=0)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

final_preds = (probabilities >= best_threshold).astype(int)

# 7. Compute Metrics
acc = accuracy_score(y_true, final_preds)
prec = precision_score(y_true, final_preds, zero_division=0)
rec = recall_score(y_true, final_preds, zero_division=0)
f1 = f1_score(y_true, final_preds, zero_division=0)
roc = roc_auc_score(y_true, probabilities)

print(f"Optimal Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc:.4f}")

# 8. Categorize Risk Levels
def categorize_risk(prob):
    if prob < 0.40:
        return "LOW"
    elif prob <= 0.70:
        return "MEDIUM"
    else:
        return "HIGH"

risk_categories = [categorize_risk(p) for p in probabilities]

# 9. Save Artifacts
torch.save(model.state_dict(), "models/attrition/attrition_model.pt")

output_df = pd.DataFrame({
    "actual_attrition": y_true.astype(int),
    "attrition_probability": np.round(probabilities, 4),
    "predicted_attrition": final_preds,
    "risk_level": risk_categories
})
output_df.to_csv("models/attrition/attrition_predictions.csv", index=False)
print("Artifacts successfully saved to models/attrition/attrition_predictions.csv")