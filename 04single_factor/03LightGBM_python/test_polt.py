# === Single-Feature SHAP: export "all point data + summary" to one Excel (LightGBM; tuned) ===
# Requirements: pip install lightgbm scikit-optimize shap openpyxl

import os, re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from scipy.stats import linregress
from lightgbm import LGBMRegressor
import shap
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- 1) Load data ----------
INPUT_PATH = r"final_Kd_database_selected.xlsx"
df = pd.read_excel(INPUT_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]
# numeric-only features (recommended for tree SHAP stability)
X = X.select_dtypes(include=[np.number]).copy()

# Features to export (must exist in X)
FEATURES = ["GATS1dv", "BCUTc-1l", "S-pH", "JGI4", "W-Salinity"]  # edit as needed

# ---------- 2) Train LightGBM (BayesSearchCV) ----------
lgbm = LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    random_state=3047,
    n_jobs=-1,
    n_estimators=500
)

param_space = {
    "n_estimators":      Integer(200, 1200),
    "num_leaves":        Integer(15, 256),
    "learning_rate":     Real(0.01, 0.2, prior="log-uniform"),
    "feature_fraction":  Real(0.5, 1.0),   # = colsample_bytree
    "bagging_fraction":  Real(0.5, 1.0),   # = subsample
    "bagging_freq":      Integer(1, 7),
    "min_child_samples": Integer(5, 100),
    "min_split_gain":    Real(0.0, 1.0),
    "reg_alpha":         Real(0.0, 1.0),
    "reg_lambda":        Real(0.0, 2.0),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

bayes = BayesSearchCV(
    estimator=lgbm,
    search_spaces=param_space,
    n_iter=48,
    cv=KFold(10, shuffle=True, random_state=3047),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=3047,
    verbose=0
)
bayes.fit(X_train, y_train)
best_model = bayes.best_estimator_

# ---------- 3) Compute SHAP (tree model) ----------
explainer = shap.TreeExplainer(best_model)
X_plot = X.copy()  # export all samples
shap_values = explainer.shap_values(X_plot)  # (n_samples, n_features) for regression

# In rare cases SHAP may return a list (e.g., multiclass); make it (n_samples, n_features)
if isinstance(shap_values, list):
    shap_values = np.mean(np.stack(shap_values, axis=0), axis=0)

# ---------- 4) Helpers ----------
def safe_sheet_name(name: str) -> str:
    # Excel sheet name ≤31 chars; forbid []:*?/\
    name = re.sub(r'[\[\]:*?/\\]', '_', name)
    return (name[:28] + '...') if len(name) > 31 else name

# ---------- 5) Export: Summary + one sheet per feature (all points) ----------
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_xlsx = os.path.join(out_dir, f"shap_single_factor_all_points_LGBM_{timestamp}.xlsx")

summary_rows = []
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    for feat in FEATURES:
        if feat not in X_plot.columns:
            summary_rows.append({"feature": feat, "note": "NOT_FOUND"})
            continue

        idx = X_plot.columns.get_loc(feat)
        x_vals = X_plot.iloc[:, idx].to_numpy()
        y_shap = shap_values[:, idx]

        # Linear regression stats: feature_value vs. SHAP value
        slope, intercept, r, p, _ = linregress(x_vals, y_shap)
        r2 = float(r**2)

        # All-point table
        df_points = pd.DataFrame({
            "sample_id": np.arange(len(x_vals)) + 1,
            "feature_value": x_vals,
            "shap_value": y_shap
        })
        df_points.to_excel(writer, sheet_name=safe_sheet_name(f"{feat}_points"), index=False)

        # Summary record
        summary_rows.append({
            "feature": feat,
            "n": int(len(x_vals)),
            "mean_abs_shap": float(np.mean(np.abs(y_shap))),
            "slope": float(slope),
            "intercept": float(intercept),
            "r": float(r),
            "R2": r2,
            "p_value": float(p),
            "x_min": float(np.min(x_vals)),
            "x_max": float(np.max(x_vals)),
            "shap_min": float(np.min(y_shap)),
            "shap_max": float(np.max(y_shap))
        })

    # Summary + BestParams sheets
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
    pd.DataFrame([bayes.best_params_]).to_excel(writer, sheet_name="BestParams", index=False)

print("All SHAP point data & stats saved to:", out_xlsx)

# ---------- 6) (Optional) Quick scatter for the first available feature ----------
first_found = next((f for f in FEATURES if f in X_plot.columns), None)
if first_found is not None:
    idx0 = X_plot.columns.get_loc(first_found)
    plt.figure(figsize=(5, 4), dpi=150)
    plt.scatter(X_plot.iloc[:, idx0], shap_values[:, idx0], s=10, alpha=0.6)
    plt.title(f"SHAP vs {first_found}")
    plt.xlabel(first_found)
    plt.ylabel("SHAP value")
    out_png = os.path.join(out_dir, f"shap_scatter_{first_found}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Quick SHAP scatter saved to:", out_png)
