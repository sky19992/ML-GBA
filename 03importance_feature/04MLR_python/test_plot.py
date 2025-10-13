# === SHAP-based Feature Importance (horizontal bar chart) — MLR version ===
# Dependencies: pip install shap openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

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

# ===== 2) Build & fit MLR pipeline =====
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

pipe.fit(X_train, y_train)
best_model = pipe  # for consistency with other scripts

# ===== 3) Importance (prefer SHAP; fallback to standardized |coef|; then PI) =====
x_label = "Mean |SHAP| value"

def _sample_rows(A, k, seed=3047):
    if len(A) <= k:
        return A.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(A), size=k, replace=False)
    return A.iloc[idx].copy()

# Background and points to explain
X_bg   = _sample_rows(X_train, k=min(500, len(X_train)))
X_plot = _sample_rows(X,       k=min(1024, len(X)))

try:
    # Use LinearExplainer on the *standardized* design matrix
    scaler = best_model.named_steps["scaler"]
    linreg = best_model.named_steps["linreg"]

    X_bg_std   = scaler.transform(X_bg)
    X_plot_std = scaler.transform(X_plot)

    explainer = shap.LinearExplainer(linreg, X_bg_std, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_plot_std)  # (n_samples, n_features)
    importance_vals = np.abs(shap_values).mean(axis=0)

except Exception:
    try:
        # Fallback 1: absolute standardized coefficients (|beta|)
        coefs = best_model.named_steps["linreg"].coef_.ravel()
        importance_vals = np.abs(coefs)
        x_label = "Absolute standardized coefficients (|β|)"
    except Exception:
        # Fallback 2: Permutation Importance (model-agnostic)
        x_label = "Permutation importance (mean |Δscore|)"
        X_pi = _sample_rows(X, k=min(2000, len(X)))
        y_pi = y.loc[X_pi.index]
        r = permutation_importance(
            best_model, X_pi, y_pi,
            n_repeats=10, random_state=3047,
            scoring="neg_mean_squared_error", n_jobs=-1
        )
        importance_vals = np.abs(r.importances_mean)

# Assemble and sort
imp_df = pd.DataFrame({"feature": X.columns, "importance": importance_vals})
imp_df = imp_df.sort_values("importance", ascending=True)

# Only show top-N
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
plt.title("Multiple Linear Regression (Standardized X)", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.25)
plt.tight_layout()

# Save figure (timestamped next to INPUT_PATH)
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(out_dir, f"importance_mlr_{timestamp}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print("Importance figure saved to:", out_png)

# ===== 5) Save importance tables =====
# Also export standardized coefficients for reference
std_coef_df = pd.DataFrame({
    "feature": X.columns,
    "std_coefficient": getattr(best_model.named_steps["linreg"], "coef_", np.array([np.nan]*X.shape[1])).ravel()
})

out_xlsx = os.path.join(out_dir, f"mlr_importance_{timestamp}.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    imp_df.sort_values("importance", ascending=False).to_excel(
        writer, sheet_name="Importance_Full", index=False
    )
    imp_top.sort_values("importance", ascending=False).to_excel(
        writer, sheet_name=f"Top_{TOP_N}", index=False
    )
    std_coef_df.sort_values("std_coefficient", key=np.abs, ascending=False).to_excel(
        writer, sheet_name="Std_Coefficients", index=False
    )

print("Tables saved to:", out_xlsx)
