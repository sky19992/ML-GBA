import os
import tempfile
from joblib import parallel_backend

# === Data & model packages ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from skopt import BayesSearchCV
from skopt.space import Real
from datetime import datetime

# Plot font
plt.rcParams["font.family"] = "Times New Roman"

# ---------------- Step 1: Load data ----------------
FILE_PATH = r"final_Kd_database_selected.xlsx"
df = pd.read_excel(FILE_PATH)

# First column = label (log Kd); others = features
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Optional: keep numeric features only
# X = X.select_dtypes(include=[np.number])

# ---------------- Step 2: Train–test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Step 3: Standardize features ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------- Step 4: ANN (MLPRegressor) + Bayesian optimization ----------------
base_ann = MLPRegressor(
    random_state=42,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1,
    solver="adam",
    shuffle=True,
    # fixed topology; you can expand search later if needed
    hidden_layer_sizes=(128, 64),
    activation="relu",
    batch_size=64
)

search_spaces = {
    "alpha": Real(1e-6, 1e-2, prior="log-uniform"),          # L2 regularization
    "learning_rate_init": Real(1e-4, 5e-2, prior="log-uniform")
}

opt = BayesSearchCV(
    estimator=base_ann,
    search_spaces=search_spaces,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_iter=24,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=0
)

with parallel_backend("loky"):
    opt.fit(X_train_scaled, y_train)

best_model = opt.best_estimator_
print("Best Parameters:", opt.best_params_)

# ---------------- Step 5: Predict ----------------
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred  = best_model.predict(X_test_scaled)

# ---------------- Step 6: Metrics ----------------
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
r2_train   = r2_score(y_train, y_train_pred)
r2_test    = r2_score(y_test,  y_test_pred)
mae_train  = mean_absolute_error(y_train, y_train_pred)
mae_test   = mean_absolute_error(y_test,  y_test_pred)
mdae_train = np.median(np.abs(y_train - y_train_pred))
mdae_test  = np.median(np.abs(y_test  - y_test_pred))

print(f"Train R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
print(f"Test  R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

# ---------------- Step 7: Plot (Actual vs Predicted) ----------------
plt.figure(figsize=(8, 7), dpi=300)
y_min = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
y_max = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())

train_scatter = plt.scatter(
    y_train, y_train_pred, c="green", label="Train",
    alpha=0.7, edgecolor="black", s=70, marker="o"
)
test_scatter = plt.scatter(
    y_test, y_test_pred, c="orange", label="Test",
    alpha=0.7, edgecolor="black", s=70, marker="o"
)

# 1:1 line
plt.plot([y_min, y_max], [y_min, y_max], color="red", linestyle="--", label="1:1 line")

# Corner metrics
plt.text(y_min, y_max*0.97, f"Train R²={r2_train:.3f}", color="green", fontsize=12)
plt.text(y_min, y_max*0.92, f"Test  R²={r2_test:.3f}",  color="orange", fontsize=12)

plt.xlabel("Actual log Kd", fontsize=16)
plt.ylabel("Predicted log Kd", fontsize=16)
plt.title("ANN (MLP): Train vs. Test", fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.grid(True, linestyle="--", alpha=0.15)
plt.xlim(y_min, y_max)
plt.ylim(y_min, y_max)
plt.tight_layout()

# Optional: save figure
plt.savefig("ann_fit_logKd.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- Step 8: Export results to Excel ----------------
train_df = pd.DataFrame({
    "row_id": X_train.index,
    "split": "train",
    "y_true": y_train.values,
    "y_pred": y_train_pred
})
train_df["residual"] = train_df["y_true"] - train_df["y_pred"]
train_df["abs_error"] = train_df["residual"].abs()

test_df = pd.DataFrame({
    "row_id": X_test.index,
    "split": "test",
    "y_true": y_test.values,
    "y_pred": y_test_pred
})
test_df["residual"] = test_df["y_true"] - test_df["y_pred"]
test_df["abs_error"] = test_df["residual"].abs()

all_df = pd.concat([train_df, test_df], axis=0).sort_values(["split", "row_id"])

metrics_df = pd.DataFrame({
    "metric": ["R2_train", "RMSE_train", "MAE_train", "MdAE_train",
               "R2_test",  "RMSE_test",  "MAE_test",  "MdAE_test"],
    "value": [r2_train, rmse_train, mae_train, mdae_train,
              r2_test,  rmse_test,  mae_test,  mdae_test]
})

best_params_df = pd.DataFrame([opt.best_params_])

out_dir = os.path.dirname(FILE_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_XLSX = os.path.join(out_dir, f"ann_predictions_{timestamp}.xlsx")

with pd.ExcelWriter(OUT_XLSX) as writer:
    train_df.to_excel(writer, sheet_name="Train", index=False)
    test_df.to_excel(writer,  sheet_name="Test",  index=False)
    all_df.to_excel(writer,   sheet_name="All",   index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    best_params_df.to_excel(writer, sheet_name="BestParams", index=False)

print(f"\n✅ All results exported to: {OUT_XLSX}")
