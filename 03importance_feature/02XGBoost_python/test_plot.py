# === SHAP-based Feature Importance (horizontal bar chart) — XGBoost version ===
# Dependencies: pip install xgboost scikit-optimize shap openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import shap
from xgboost import XGBRegressor
from datetime import datetime

plt.rcParams["font.family"] = "Times New Roman"

# ===== 1) Load Excel =====
INPUT_PATH = r"final_Kd_database_selected.xlsx"
df = pd.read_excel(INPUT_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]
# If there are non-numeric columns, uncomment:
# X = X.select_dtypes(include=[np.number])

# ===== 2) Train XGBoost (BayesSearchCV) =====
xgb = XGBRegressor(
    objective="reg:squarederror",
    booster="gbtree",
    tree_method="hist",
    random_state=3047,
    n_jobs=-1
)

param_space = {
    "n_estimators":     Integer(200, 700),
    "max_depth":        Integer(3, 12),
    "learning_rate":    Real(0.01, 0.2, prior="log-uniform"),
    "subsample":        Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
    "min_child_weight": Integer(1, 10),
    "reg_alpha":        Real(0.0, 1.0),    # L1
    "reg_lambda":       Real(0.0, 2.0),    # L2
    "gamma":            Real(0.0, 5.0)     # Minimum loss reduction to split
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

bayes = BayesSearchCV(
    estimator=xgb,
    search_spaces=param_space,
    n_iter=48,                                  # increase for a more thorough search
    cv=KFold(10, shuffle=True, random_state=3047),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=3047,
    verbose=0
)
bayes.fit(X_train, y_train)
best_model = bayes.best_estimator_

# ===== 3) Compute importance (prefer SHAP; fallback to built-in) =====
X_plot = X_test
x_label = "Mean |SHAP| value"

try:
    # TreeExplainer is fast and stable for tree models
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_plot)   # shape: (n_samples, n_features)
    importance_vals = np.abs(shap_values).mean(axis=0)
except Exception:
    # Fallback: XGB built-in importance
    importance_vals = getattr(best_model, "feature_importances_", None)
    if importance_vals is None or np.allclose(importance_vals, 0):
        importance_vals = np.zeros(X_plot.shape[1], dtype=float)
    x_label = "Feature importance (XGBoost built-in)"

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
plt.xlabel(x_label, fontsize=12)
plt.ylabel("")
plt.title("XGBoost", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.25)
plt.tight_layout()

# Save figure (timestamped, next to INPUT_PATH)
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(out_dir, f"shap_importance_xgb_{timestamp}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print("Importance figure saved to:", out_png)

# ===== 5) Save importance tables & best params =====
out_xlsx = os.path.join(out_dir, f"xgb_importance_{timestamp}.xlsx")
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
