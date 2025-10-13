# === Single-Feature SHAP: export "all point data + summary" to one Excel (Random Forest) ===
# Requirements: pip install scikit-optimize shap openpyxl

import os
import re
import inspect
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from scipy.stats import linregress
import shap
from datetime import datetime

# ---------- Basic settings ----------
# Save next to the INPUT file (more convenient when running from notebooks/IDE)
# If you prefer the script's directory, replace out_dir with SCRIPT_DIR.
SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# ---------- 1) Load data ----------
INPUT_PATH = r"final_Kd_database_selected.xlsx"
df = pd.read_excel(INPUT_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]

# Ensure numeric features only
X = X.select_dtypes(include=[np.number]).copy()

# Features to export (must match X column names)
FEATURES = ["GATS1dv", "BCUTc-1l", "S-pH", "JGI4", "W-Salinity"]  # edit as needed

# ---------- 2) Train RandomForest (BayesSearchCV) ----------
rf = RandomForestRegressor(random_state=3047, n_jobs=-1, bootstrap=True)
param_space = {
    "n_estimators":      Integer(200, 700),
    "max_depth":         Integer(3, 35),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf":  Integer(1, 10),
    "max_features":      Real(0.1, 1.0),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

bayes = BayesSearchCV(
    estimator=rf,
    search_spaces=param_space,
    n_iter=32,
    cv=KFold(5, shuffle=True, random_state=3047),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=3047,
    verbose=0
)
bayes.fit(X_train, y_train)
best_model = bayes.best_estimator_

# ---------- 3) Compute SHAP (tree-based model) ----------
explainer = shap.TreeExplainer(best_model)
X_plot = X.copy()  # export all samples

shap_values = explainer.shap_values(X_plot)  # (n_samples, n_features) for regression
# Handle possible list return shapes (classification); keep regression behavior robust
if isinstance(shap_values, list):
    # take the mean across classes (rare for regression; just in case)
    shap_values = np.mean(np.stack(shap_values, axis=0), axis=0)

# ---------- 4) Write Excel: Summary + one sheet per feature (all point data) ----------
def safe_sheet_name(name: str) -> str:
    # Excel sheet name must be ≤31 chars and not contain []:*?/\
    name = re.sub(r'[\[\]:*?/\\]', '_', name)
    return (name[:28] + '...') if len(name) > 31 else name

out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_xlsx = os.path.join(out_dir, f"shap_single_factor_all_points_RF_{timestamp}.xlsx")

summary_rows = []
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    for feat in FEATURES:
        if feat not in X_plot.columns:
            summary_rows.append({"feature": feat, "note": "NOT_FOUND"})
            continue

        idx = X_plot.columns.get_loc(feat)
        x_vals = X_plot.iloc[:, idx].to_numpy()
        y_shap = shap_values[:, idx]

        # Linear regression stats (feature vs. SHAP value)
        slope, intercept, r, p, _ = linregress(x_vals, y_shap)
        r2 = float(r ** 2)

        # All-point table for this feature
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
            "shap_max": float(np.max(y_shap)),
        })

    # Summary sheet (present features first, then any not-found entries)
    summary_df = pd.DataFrame(summary_rows)
    # Optional: move 'note' to the end if present
    cols = [c for c in summary_df.columns if c != "note"] + ([ "note"] if "note" in summary_df.columns else [])
    summary_df = summary_df[cols]
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

print("All SHAP point data & stats saved to:", out_xlsx)
