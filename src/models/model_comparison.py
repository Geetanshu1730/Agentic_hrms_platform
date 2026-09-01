import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.makedirs("models/attrition", exist_ok=True)

# 1. Load ML-Ready Datasets
X_train_df = pd.read_csv("data/processed/ml_ready/attrition_X_train.csv")
X_test_df = pd.read_csv("data/processed/ml_ready/attrition_X_test.csv")
y_train = pd.read_csv("data/processed/ml_ready/attrition_y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/ml_ready/attrition_y_test.csv").values.ravel()

results = []

# --- Model 1: Logistic Regression (Baseline) ---
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_df, y_train)
lr_probs = lr.predict_proba(X_test_df)[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

results.append({
    "Algorithm": "Logistic Regression (Linear Baseline)",
    "Accuracy": round(accuracy_score(y_test, lr_preds), 4),
    "Precision": round(precision_score(y_test, lr_preds, zero_division=0), 4),
    "Recall": round(recall_score(y_test, lr_preds, zero_division=0), 4),
    "F1-Score": round(f1_score(y_test, lr_preds, zero_division=0), 4),
    "ROC-AUC": round(roc_auc_score(y_test, lr_probs), 4),
    "Architecture Type": "Linear Statistical"
})

# --- Model 2: Random Forest (Ensemble Trees) ---
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_df, y_train)
rf_probs = rf.predict_proba(X_test_df)[:, 1]
rf_preds = (rf_probs >= 0.5).astype(int)

results.append({
    "Algorithm": "Random Forest (Tree Ensemble)",
    "Accuracy": round(accuracy_score(y_test, rf_preds), 4),
    "Precision": round(precision_score(y_test, rf_preds, zero_division=0), 4),
    "Recall": round(recall_score(y_test, rf_preds, zero_division=0), 4),
    "F1-Score": round(f1_score(y_test, rf_preds, zero_division=0), 4),
    "ROC-AUC": round(roc_auc_score(y_test, rf_probs), 4),
    "Architecture Type": "Non-Linear Bagging"
})

# --- Model 3: PyTorch Deep Neural Network (Platform Engine) ---
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

dnn = AttritionNetwork(input_dim=X_train_df.shape[1])
if os.path.exists("models/attrition/attrition_model.pt"):
    dnn.load_state_dict(torch.load("models/attrition/attrition_model.pt"))
dnn.eval()

X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
with torch.no_grad():
    dnn_probs = torch.sigmoid(dnn(X_test_tensor)).numpy().flatten()
dnn_preds = (dnn_probs >= 0.5).astype(int)

results.append({
    "Algorithm": "Deep Neural Network (PyTorch MLP)",
    "Accuracy": round(accuracy_score(y_test, dnn_preds), 4),
    "Precision": round(precision_score(y_test, dnn_preds, zero_division=0), 4),
    "Recall": round(recall_score(y_test, dnn_preds, zero_division=0), 4),
    "F1-Score": round(f1_score(y_test, dnn_preds, zero_division=0), 4),
    "ROC-AUC": round(roc_auc_score(y_test, dnn_probs), 4),
    "Architecture Type": "Multi-Layer Perceptron"
})

df_benchmarks = pd.DataFrame(results)
df_benchmarks.to_csv("models/attrition/model_benchmarks.csv", index=False)
print("Benchmark comparison table saved to models/attrition/model_benchmarks.csv")
print(df_benchmarks)