import os
import tempfile
from joblib import parallel_backend
from datetime import datetime


# === Data & model packages ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from skopt import BayesSearchCV
from skopt.space import Real

# Plot font
plt.rcParams["font.family"] = "Times New Roman"

# ---------------- Step 1: Load Data ----------------
FILE_PATH = "final_Kd_database_selected.xlsx"
df = pd.read_excel(FILE_PATH)

# First column = label (log Kd); others = features
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Optional: keep numeric features only
# X = X.select_dtypes(include=[np.number])

# ---------------- Step 2: Train–Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

# ---------------- Step 3: Feature Standardization ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------- Step 4: SVR + Bayesian Optimization ----------------
search_spaces = {
    "C":       Real(10.0, 1000.0, prior="log-uniform"),
    "epsilon": Real(0.02, 0.2,    prior="log-uniform"),
    "gamma":   Real(1e-3, 1e-1,   prior="log-uniform"),
}

opt = BayesSearchCV(
    estimator=SVR(kernel="rbf"),
    search_spaces=search_spaces,
    cv=KFold(n_splits=5, shuffle=True, random_state=3047),
    n_iter=32,
    scoring="r2",
    n_jobs=-1,
    random_state=3047,
    verbose=0,
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

print(f"Train R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
print(f"Test  R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

# === 6) 5-fold cross-validation on train data with best model ===
cv = KFold(5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
print("\n5-fold CV R²:", np.round(cv_scores, 4),
      "Mean =", round(cv_scores.mean(), 4), "Std =", round(cv_scores.std(), 4))


# ---------------- Step 7: Plot (Train vs Test) ----------------
plt.figure(figsize=(8, 7), dpi=300)

# Scatter
plt.scatter(y_train, y_train_pred, color="#2ca02c", label="Train", alpha=0.7, edgecolor="black", s=60)
plt.scatter(y_test,  y_test_pred,  color="#ff7f0e", label="Test",  alpha=0.7, edgecolor="black", s=60)

# 45° reference line using global min/max
global_min = min(y.min(), y_train_pred.min(), y_test_pred.min())
global_max = max(y.max(), y_train_pred.max(), y_test_pred.max())
plt.plot([global_min, global_max], [global_min, global_max], "--", color="gray", label="1:1 line")

# Corner metrics (concise)
plt.text(global_min, global_max*0.97, f"Train R²={r2_train:.3f}", color="#2ca02c", fontsize=12)
plt.text(global_min, global_max*0.92, f"Test  R²={r2_test:.3f}",  color="#ff7f0e", fontsize=12)

plt.xlabel("Actual log Kd", fontsize=16)
plt.ylabel("Predicted log Kd", fontsize=16)
plt.title("SVR (RBF): Train vs. Test", fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.grid(True, linestyle="--", alpha=0.15)
plt.tight_layout()

# Save figure (optional)
plt.savefig("svr_fit_logKd.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- (Optional) Export predictions to Excel ----------------
out_dir = os.path.dirname(FILE_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_xlsx = os.path.join(out_dir, f"svr_predictions_{timestamp}.xlsx")
pd.DataFrame({
     "row_id": np.r_[X_train.index, X_test.index],
     "split":  ["train"]*len(X_train) + ["test"]*len(X_test),
     "y_true": np.r_[y_train.values, y_test.values],
     "y_pred": np.r_[y_train_pred,  y_test_pred],
 }).to_excel(out_xlsx, index=False)
print(f"Predictions exported to: {out_xlsx}")
