# === SHAP-based Feature Importance (horizontal bar chart) — LightGBM version ===
# Dependencies: pip install lightgbm shap scikit-optimize openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from lightgbm import LGBMRegressor
import shap
from datetime import datetime

# ---- Avoid joblib temp-path issues on Windows (ASCII-only dir) ----
TMP_DIR = r"C:\joblib_temp"
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = TMP_DIR

plt.rcParams["font.family"] = "Times New Roman"

# ===== 1) Load Excel =====
INPUT_PATH = r"F:/Kd_process/machine_learning/RFECV_plot/final_Kd_database_selected.xlsx"
df = pd.read_excel(INPUT_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]
# If there are non-numeric columns, uncomment:
# X = X.select_dtypes(include=[np.number])

# ===== 2) Train LightGBM (BayesSearchCV) =====
lgbm = LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    random_state=3047,
    n_jobs=-1,
    n_estimators=500
)

param_space = {
    "n_estimators":        Integer(200, 700),
    "num_leaves":          Integer(15, 256),
    "learning_rate":       Real(0.01, 0.2, prior="log-uniform"),
    "feature_fraction":    Real(0.5, 1.0),   # = colsample_bytree
    "bagging_fraction":    Real(0.5, 1.0),   # = subsample
    "bagging_freq":        Integer(1, 7),
    "min_child_samples":   Integer(5, 100),
    "min_split_gain":      Real(0.0, 1.0),
    "reg_alpha":           Real(0.0, 1.0),
    "reg_lambda":          Real(0.0, 2.0),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

bayes = BayesSearchCV(
    estimator=lgbm,
    search_spaces=param_space,
    n_iter=48,
    cv=KFold(5, shuffle=True, random_state=3047),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=3047,
    verbose=0
)
bayes.fit(X_train, y_train)
best_model = bayes.best_estimator_

# ===== 3) Compute importance (prefer SHAP; fallback to gain) =====
X_plot = X  # Use full data to estimate global importance (optionally use X_test)
label_for_x = "Mean |SHAP| value"

try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_plot)
    # For regression: shap_values shape = (n_samples, n_features)
    # For multiclass classification: list/3D; but here it's regression.
    importance_vals = np.abs(shap_values).mean(axis=0)
except Exception:
    # Fallback: LightGBM gain importance aligned to column names
    booster = best_model.booster_
    gain_vals = booster.feature_importance(importance_type="gain")
    feat_names = booster.feature_name()
    gain_map = dict(zip(feat_names, gain_vals))
    importance_vals = np.array([gain_map.get(str(c), 0.0) for c in X_plot.columns])
    label_for_x = "Feature importance (gain)"

imp_df = pd.DataFrame({"feature": X_plot.columns, "importance": importance_vals})
imp_df = imp_df.sort_values("importance", ascending=True)

# Show top-N
TOP_N = min(12, len(imp_df))
imp_top = imp_df.tail(TOP_N)

# ===== 4) Plot =====
plt.figure(figsize=(7.2, 5.6), dpi=300)
plt.barh(
    imp_top["feature"], imp_top["importance"],
    color="#6B88C8", edgecolor="white", linewidth=0.7
)
plt.xlabel(label_for_x, fontsize=12)
plt.ylabel("")
plt.title("LightGBM", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.25)
plt.tight_layout()

# Save figure
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(out_dir, f"shap_importance_lgbm_{timestamp}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print("Importance figure saved to:", out_png)

# ===== 5) Save importance tables & best params =====
out_xlsx = os.path.join(out_dir, f"lgbm_importance_{timestamp}.xlsx")
with pd.ExcelWriter(out_xlsx) as writer:
    imp_df.sort_values("importance", ascending=False).to_excel(
        writer, sheet_name="Importance_Full", index=False
    )
    imp_top.sort_values("importance", ascending=False).to_excel(
        writer, sheet_name=f"Top_{TOP_N}", index=False
    )
    pd.DataFrame([bayes.best_params_]).to_excel(
        writer, sheet_name="BestParams", index=False
    )

print("Tables saved to:", out_xlsx)
