# === SHAP-based Feature Importance (horizontal bar chart) — SVR + Pipeline ===
# Dependencies: pip install scikit-optimize shap openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

from skopt import BayesSearchCV
from skopt.space import Real
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

# ===== 2) Build SVR pipeline =====
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

# NOTE: When tuning a Pipeline, prefix parameters with the step name, e.g. svr__*
search_spaces = {
    "svr__C":       Real(10.0, 1000.0, prior="log-uniform"),
    "svr__epsilon": Real(0.02, 0.2,    prior="log-uniform"),
    "svr__gamma":   Real(1e-3, 1e-1,   prior="log-uniform"),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    n_iter=32,
    cv=KFold(5, shuffle=True, random_state=3047),
    scoring="neg_mean_squared_error",  # switch to "r2" if preferred
    n_jobs=-1,
    random_state=42,
    verbose=0
)
opt.fit(X_train, y_train)
best_model = opt.best_estimator_

# ===== 3) Compute importance (prefer SHAP; fallback to permutation importance) =====
x_label = "Mean |SHAP| value"

# Helper: sample rows to keep Kernel SHAP tractable
def _sample_rows(A, k, seed=3047):
    if len(A) <= k:
        return A.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(A), size=k, replace=False)
    return A.iloc[idx].copy()

# Background set for Kernel SHAP and data to explain
X_bg   = _sample_rows(X_train, k=min(200,  len(X_train)))  # background
X_plot = _sample_rows(X,       k=min(512,  len(X)))        # points to explain (or use X_test)

try:
    # Robust predictor wrapper (works with Pipeline)
    def f(A):
        if isinstance(A, np.ndarray):
            A = pd.DataFrame(A, columns=X.columns)
        return best_model.predict(A)

    # Kernel SHAP (model-agnostic); background helps speed & stability
    explainer = shap.Explainer(f, X_bg, algorithm="kernel")
    shap_values = explainer(X_plot).values  # shape: (n_samples, n_features)
    importance_vals = np.abs(shap_values).mean(axis=0)

except Exception:
    # Fallback: model-agnostic permutation importance (stable across versions)
    x_label = "Permutation importance (mean |Δscore|)"
    X_pi = _sample_rows(X, k=min(2000, len(X)))
    y_pi = y.loc[X_pi.index]
    result = permutation_importance(
        best_model, X_pi, y_pi,
        n_repeats=10, random_state=3047,
        scoring="neg_mean_squared_error", n_jobs=-1
    )
    importance_vals = np.abs(result.importances_mean)

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
plt.title("SVR (RBF) + StandardScaler", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.25)
plt.tight_layout()

# Save figure (timestamped, next to INPUT_PATH)
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(out_dir, f"importance_svr_pipe_{timestamp}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print("Importance figure saved to:", out_png)

# ===== 5) Save importance tables & best params =====
out_xlsx = os.path.join(out_dir, f"svr_pipe_importance_{timestamp}.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    imp_df.sort_values("importance", ascending=False).to_excel(
        writer, sheet_name="Importance_Full", index=False
    )
    imp_top.sort_values("importance", ascending=False).to_excel(
        writer, sheet_name=f"Top_{TOP_N}", index=False
    )
    pd.DataFrame([opt.best_params_]).to_excel(
        writer, sheet_name="BestParams", index=False
    )

print("Tables saved to:", out_xlsx)
