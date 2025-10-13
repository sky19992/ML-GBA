# === SHAP-based Feature Importance (horizontal bar chart) — RandomForest version ===
# Dependencies: pip install scikit-optimize shap openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import shap
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

# ===== 2) Train RandomForest (BayesSearchCV) =====
rf = RandomForestRegressor(
    random_state=3047,
    n_jobs=-1,
    bootstrap=True
)

param_space = {
    "n_estimators":      Integer(200, 700),
    "max_depth":         Integer(3, 35),
    "min_samples_split": Integer(2, 20),
    "min_samples_leaf":  Integer(1, 10),
    "max_features":      Real(0.1, 1.0)
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

# ===== 3) Compute importance (prefer SHAP; fallback to RF Gini) =====
X_plot = X  # Use full data to estimate global importance (or switch to X_test)
x_label = "Mean |SHAP| value"

try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_plot)  # (n_samples, n_features) for regression
    importance_vals = np.abs(shap_values).mean(axis=0)
except Exception:
    importance_vals = best_model.feature_importances_
    x_label = "Feature importance (Gini)"

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
plt.title("Random Forest", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.25)
plt.tight_layout()

# Save figure (timestamped, next to INPUT_PATH)
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(out_dir, f"shap_importance_rf_{timestamp}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print("Importance figure saved to:", out_png)

# ===== 5) Save importance tables & best params =====
out_xlsx = os.path.join(out_dir, f"rf_importance_{timestamp}.xlsx")
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
