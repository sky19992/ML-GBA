# === Multiple Linear Regression (OLS) full pipeline ===
# Table description: The first column is the label (log Kd), and the other columns are features.
# The first row contains the feature names. Input file must be .xlsx.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV

# Stats (optional but useful)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---- Optional: avoid joblib temp path issues on Windows non-ASCII paths ----
TMP_DIR = r"C:\joblib_temp"   # ensure ASCII-only directory exists
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = TMP_DIR

# Toggle Ridge (L2) instead of plain OLS
USE_RIDGE = False

# === 1. Provide the input file path ===
INPUT_PATH = r"H:/Kd_process/machine_learning/RFECV_plot/final_Kd_database_selected.xlsx"

# === 2. Read data ===
df = pd.read_excel(INPUT_PATH)
target_col = df.columns[0]

# Keep numeric features only; if you have categoricals, one-hot encode before running
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
y = df[target_col].values

# Drop all-NaN or zero-variance columns (avoid singular matrices)
X = X.dropna(axis=1, how="all")
const_mask = X.std(axis=0, skipna=True) > 0
X = X.loc[:, const_mask]

# Fill remaining NaNs with column means (OLS requires no missing values)
X = X.fillna(X.mean())

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4. Define model (with scaling) ===
if USE_RIDGE:
    # Ridge + standardization + CV-selected alpha
    alphas = np.logspace(-4, 4, 50)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", RidgeCV(alphas=alphas, store_cv_values=True))
    ])
    model_name = "Ridge Regression (with standardization)"
else:
    # Classic OLS: standardize features then linear regression
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", LinearRegression())
    ])
    model_name = "Multiple Linear Regression (OLS, standardized features)"

# === 5. Fit model ===
model.fit(X_train, y_train)

# === 6. Evaluate performance ===
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
mae_train  = mean_absolute_error(y_train, y_pred_train)
mae_test   = mean_absolute_error(y_test,  y_pred_test)
mdae_train = np.median(np.abs(y_train - y_pred_train))
mdae_test  = np.median(np.abs(y_test  - y_pred_test))

print(f"\nModel: {model_name}")
if USE_RIDGE:
    print("Chosen alpha (Ridge):", model.named_steps["reg"].alpha_)

print("\nModel performance:")
print(f" Training R² = {r2_score(y_train, y_pred_train):.4f}, RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}, MdAE = {mdae_train:.4f}")
print(f" Testing  R² = {r2_score(y_test,  y_pred_test):.4f}, RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f},  MdAE = {mdae_test:.4f}")

# === 7. 5-fold cross-validation on train data ===
cv = KFold(5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
print("\n5-fold CV R²:", np.round(cv_scores, 4),
      "Mean =", round(cv_scores.mean(), 4), "Std =", round(cv_scores.std(), 4))

# === 8. Scatter plot: True vs Pred ===
plt.figure(figsize=(6, 6), dpi=300)
sns.scatterplot(x=y_train, y=y_pred_train, label="Train", alpha=0.6)
sns.scatterplot(x=y_test,  y=y_pred_test,  label="Test",  alpha=0.6)
min_, max_ = min(y.min(), y_pred_train.min(), y_pred_test.min()), max(y.max(), y_pred_train.max(), y_pred_test.max())
plt.plot([min_, max_], [min_, max_], "--", color="gray")
plt.text(min_, max_*0.95, f"Train R²={r2_score(y_train, y_pred_train):.3f}")
plt.text(min_, max_*0.90, f"Test  R²={r2_score(y_test,  y_pred_test):.3f}")
plt.xlabel("True log Kd")
plt.ylabel("Predicted log Kd")
plt.title(f"{model_name}: Train vs. Test Fit")
plt.legend()
plt.tight_layout()
plt.show()

# === 9. Export predictions & metrics to Excel ===
from datetime import datetime

train_df = pd.DataFrame({
    "row_id": X_train.index,
    "split": "train",
    "y_true": y_train,
    "y_pred": y_pred_train
})
train_df["residual"] = train_df["y_true"] - train_df["y_pred"]
train_df["abs_error"] = train_df["residual"].abs()

test_df = pd.DataFrame({
    "row_id": X_test.index,
    "split": "test",
    "y_true": y_test,
    "y_pred": y_pred_test
})
test_df["residual"] = test_df["y_true"] - test_df["y_pred"]
test_df["abs_error"] = test_df["residual"].abs()

all_df = pd.concat([train_df, test_df], axis=0).sort_values(["split", "row_id"])

metrics_df = pd.DataFrame({
    "metric": ["R2_train", "RMSE_train", "MAE_train", "MdAE_train",
               "R2_test",  "RMSE_test",  "MAE_test",  "MdAE_test",
               "CV_R2_mean", "CV_R2_std"],
    "value": [r2_score(y_train, y_pred_train), rmse_train, mae_train, mdae_train,
              r2_score(y_test, y_pred_test),  rmse_test,  mae_test,  mdae_test,
              cv_scores.mean(), cv_scores.std()]
})

# === 10. Coefficients table (with standardized X) ===
# Get standardized design matrix from the pipeline
scaler = model.named_steps["scaler"]
X_train_std = pd.DataFrame(
    scaler.transform(X_train),
    index=X_train.index,
    columns=X_train.columns
)

# Coefficients / intercept from the fitted regressor
coef = model.named_steps["reg"].coef_
intercept = model.named_steps["reg"].intercept_

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": coef
}).sort_values("coefficient", key=np.abs, ascending=False)

# === 11. p-values via statsmodels OLS (with constant) ===
X_sm = sm.add_constant(X_train_std)
ols_res = sm.OLS(y_train, X_sm).fit()
pvals = ols_res.pvalues.rename("p_value")
coef_sm = ols_res.params.rename("coef_sm")
coef_full = pd.concat([coef_sm, pvals], axis=1).reset_index().rename(columns={"index":"feature"})
coef_full = coef_full[coef_full["feature"] != "const"]  # keep only feature rows

# === 12. VIF (on standardized X_train) ===
vif_df = pd.DataFrame({
    "feature": X_train_std.columns,
    "VIF": [variance_inflation_factor(X_train_std.values, i) for i in range(X_train_std.shape[1])]
}).sort_values("VIF", ascending=False)

# === 13. Save all ===
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_XLSX = os.path.join(out_dir, f"mlr_predictions_{timestamp}.xlsx")

with pd.ExcelWriter(OUT_XLSX) as writer:
    train_df.to_excel(writer, sheet_name="Train", index=False)
    test_df.to_excel(writer,  sheet_name="Test",  index=False)
    all_df.to_excel(writer,   sheet_name="All",   index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    coef_df.to_excel(writer, sheet_name="Coef_sorted_abs", index=False)
    coef_full.to_excel(writer, sheet_name="OLS_coef_pvalues", index=False)
    vif_df.to_excel(writer, sheet_name="VIF", index=False)
    X_train_std.to_excel(writer, sheet_name="X_train_std", index=True)

print(f"\nPredictions exported to: {OUT_XLSX}")
if USE_RIDGE:
    print("Note: Ridge regression used. See 'Metrics' and 'Coef_sorted_abs' sheets.")
