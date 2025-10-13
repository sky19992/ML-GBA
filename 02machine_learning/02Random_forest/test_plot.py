# === Imports ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from datetime import datetime

# === 1) Input path ===
INPUT_PATH = r"final_Kd_database_selected.xlsx"

# === 2) Read data ===
df = pd.read_excel(INPUT_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]

# === 3) Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

# === 4) Standardize features ===
# Note: Random Forests don't require scaling, but we keep it for consistency with other models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
feature_names = X.columns.tolist()

# (Optional) keep DataFrame wrappers for readability downstream
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
X_test_df  = pd.DataFrame(X_test_scaled,  columns=feature_names, index=X_test.index)

# === 5) Random Forest + Bayesian Optimization ===
# max_features: float in (0, 1] means a fraction of features; valid for sklearn RF.
param_space = {
    "n_estimators":      Integer(200, 700),
    "max_depth":         Integer(3, 35),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf":  Integer(1, 10),
    "max_features":      Real(0.1, 1.0)
}

rf = RandomForestRegressor(random_state=3047)
bayes = BayesSearchCV(
    estimator=rf,
    search_spaces=param_space,
    n_iter=32,
    cv=KFold(5, shuffle=True, random_state=3047),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=3047
)

print("Starting Bayesian optimization...")
bayes.fit(X_train_scaled, y_train)
best_model = bayes.best_estimator_

# === 6) Evaluate performance ===
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test  = best_model.predict(X_test_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
mae_train  = mean_absolute_error(y_train, y_pred_train)
mae_test   = mean_absolute_error(y_test,  y_pred_test)
mdae_train = np.median(np.abs(y_train - y_pred_train))
mdae_test  = np.median(np.abs(y_test  - y_pred_test))

print("\nBest parameters:")
print(bayes.best_params_)

print("\nModel performance:")
print(f" Training R² = {r2_score(y_train, y_pred_train):.4f}, RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}, MdAE = {mdae_train:.4f}")
print(f" Testing  R² = {r2_score(y_test,  y_pred_test):.4f}, RMSE = {rmse_test:.4f},  MAE = {mae_test:.4f},  MdAE = {mdae_test:.4f}")

# === 7) 10-fold cross-validation on the whole dataset ===
cv = KFold(10, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_model, scaler.transform(X), y,
    cv=cv, scoring="r2", n_jobs=-1
)
print("\n10-fold CV R²:", np.round(cv_scores, 4),
      "Mean =", round(cv_scores.mean(), 4), "Std =", round(cv_scores.std(), 4))

# === 8) Fit scatter plot (Train vs Test) ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_train, y=y_pred_train, label="Train", color="blue", alpha=0.6)
sns.scatterplot(x=y_test,  y=y_pred_test,  label="Test",  color="red",  alpha=0.6)

min_, max_ = min(y.min(), y_pred_train.min(), y_pred_test.min()), max(y.max(), y_pred_train.max(), y_pred_test.max())
plt.plot([min_, max_], [min_, max_], "--", color="gray")
plt.text(min_, max_*0.95, f"Train R²={r2_score(y_train, y_pred_train):.3f}", color="blue")
plt.text(min_, max_*0.90, f"Test  R²={r2_score(y_test,  y_pred_test):.3f}", color="red")

plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("Train vs. Test Fit")
plt.legend()
plt.tight_layout()
plt.show()

# === 9) Export predictions & metrics to Excel ===
# Assemble result tables (preserve original row index for traceability)
train_df = pd.DataFrame({
    "row_id": X_train.index,
    "split": "train",
    "y_true": y_train.values,
    "y_pred": y_pred_train
})
train_df["residual"] = train_df["y_true"] - train_df["y_pred"]
train_df["abs_error"] = train_df["residual"].abs()

test_df = pd.DataFrame({
    "row_id": X_test.index,
    "split": "test",
    "y_true": y_test.values,
    "y_pred": y_pred_test
})
test_df["residual"] = test_df["y_true"] - test_df["y_pred"]
test_df["abs_error"] = test_df["residual"].abs()

all_df = pd.concat([train_df, test_df], axis=0).sort_values(["split", "row_id"])

# Metrics summary
metrics_df = pd.DataFrame({
    "metric": ["R2_train", "RMSE_train", "MAE_train", "MdAE_train",
               "R2_test",  "RMSE_test",  "MAE_test",  "MdAE_test",
               "CV_R2_mean", "CV_R2_std"],
    "value": [r2_score(y_train, y_pred_train), rmse_train, mae_train, mdae_train,
              r2_score(y_test, y_pred_test),  rmse_test,  mae_test,  mdae_test,
              cv_scores.mean(), cv_scores.std()]
})

best_params_df = pd.DataFrame([bayes.best_params_])

# Output path (same directory as input)
out_dir = os.path.dirname(INPUT_PATH)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_XLSX = os.path.join(out_dir, f"rf_predictions_{timestamp}.xlsx")

with pd.ExcelWriter(OUT_XLSX) as writer:
    train_df.to_excel(writer, sheet_name="Train", index=False)
    test_df.to_excel(writer,  sheet_name="Test",  index=False)
    all_df.to_excel(writer,   sheet_name="All",   index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    best_params_df.to_excel(writer, sheet_name="BestParams", index=False)

print(f"\nPredictions exported to: {OUT_XLSX}")
