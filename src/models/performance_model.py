import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Ensure target directory exists
os.makedirs("models/performance", exist_ok=True)

# 2. Load ML-Ready Performance Data
X_train_df = pd.read_csv("data/processed/ml_ready/performance_X_train.csv")
X_test_df = pd.read_csv("data/processed/ml_ready/performance_X_test.csv")
y_train_df = pd.read_csv("data/processed/ml_ready/performance_y_train.csv")
y_test_df = pd.read_csv("data/processed/ml_ready/performance_y_test.csv")

input_dim = X_train_df.shape[1]
print(f"Loaded Performance Dataset: {input_dim} features.")
print(f"Training Samples: {X_train_df.shape[0]}, Test Samples: {X_test_df.shape[0]}")

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
y_train = torch.tensor(y_train_df.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
y_test = torch.tensor(y_test_df.values, dtype=torch.float32).view(-1, 1)

# 3. Model Architecture for Performance Regression
class PerformanceNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PerformanceNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# 4. Training Setup
torch.manual_seed(42)
np.random.seed(42)

model = PerformanceNetwork(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

# 5. Training Loop
batch_size = 32
num_epochs = 60
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

# 6. Evaluation on Test Set
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy().flatten()

y_true = y_test.numpy().flatten()

mse = mean_squared_error(y_true, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, preds)
r2 = r2_score(y_true, preds)

print("\n--- PERFORMANCE MODEL EVALUATION ---")
print(f"R2 Score:  {r2:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"MAE:       {mae:.4f}")

# 7. Tier Categorization (High / Meets / Needs Improvement)
p25, p75 = np.percentile(preds, 25), np.percentile(preds, 75)

def categorize_performance(score):
    if score >= p75:
        return "HIGH_PERFORMER"
    elif score >= p25:
        return "MEETS_EXPECTATIONS"
    else:
        return "NEEDS_IMPROVEMENT"

performance_tiers = [categorize_performance(s) for s in preds]

# 8. Save Model & Predictions Artifacts
torch.save(model.state_dict(), "models/performance/performance_model.pt")

output_df = pd.DataFrame({
    "actual_score": y_true,
    "predicted_score": np.round(preds, 3),
    "performance_tier": performance_tiers
})
output_df.to_csv("models/performance/performance_predictions.csv", index=False)
print("\nModel saved to models/performance/performance_model.pt")
print("Predictions saved to models/performance/performance_predictions.csv")