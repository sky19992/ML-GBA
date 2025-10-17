# === SHAP-based Feature Importance (horizontal bar chart) — ANN (MLP) version ===
# Dependencies: pip install scikit-optimize shap openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

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

# ===== 2) Build ANN pipeline (fixed topology; tune alpha & lr_init) =====
base_ann = MLPRegressor(
    random_state=42,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1,
    solver="adam",
    shuffle=True,
    hidden_layer_sizes=(128, 64),
    activation="relu",
    batch_size=64,
    # These will be overridden by BayesSearchCV
    alpha=1e-4,
    learning_rate_init=1e-3,
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", base_ann),
])

search_spaces = {
    "mlp__alpha": Real(1e-6, 1e-2, prior="log-uniform"),
    "mlp__learning_rate_init": Real(1e-4, 5e-2, prior="log-uniform"),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3047
)

opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_iter=24,
    scoring="r2",
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
opt.fit(X_train, y_train)
best_model = opt.best_estimator_

# ===== 3) Compute importance (prefer SHAP; fallback to permutation importance) =====
x_label = "Mean |SHAP| value"

def _sample_rows(A, k, seed=3047):
    if len(A) <= k:
        return A.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(A), size=k, replace=False)
    return A.iloc[idx].copy()

# Kernel SHAP with background + explanation subsamples for tractability
X_bg   = _sample_rows(X_train, k=min(200, len(X_train)))   # background
X_plot = _sample_rows(X_test,       k=min(512, len(X)))         # points to explain (or use X_test)

try:
    # Wrapper to handle DataFrame/ndarray inputs through the Pipeline
    def f(A):
        if isinstance(A, np.ndarray):
            A = pd.DataFrame(A, columns=X.columns)
        return best_model.predict(A)

    explainer = shap.Explainer(f, X_bg, algorithm="kernel")
    shap_values = explainer(X_plot).values  # (n_samples, n_features)
    importance_vals = np.abs(shap_values).mean(axis=0)

except Exception:
    # Fallback: model-agnostic permutation importance
    x_label = "Permutation importance (mean |Δscore|)"
    scorer = get_scorer("neg_mean_squared_error")
    X_pi = _sample_rows(X, k=min(2000, len(X)))
    y_pi = y.loc[X_pi.index]
    r = permutation_importance(
        best_model, X_pi, y_pi,
        n_repeats=10, random_state=3047,
        scoring="neg_mean_squared_error", n_jobs=-1
    )
    importance_vals = np.abs(r.importances_mean)

# ===== 4) Summarize & plot =====
imp_df = pd.DataFrame({"feature": X.columns, "importance": importance_vals})
imp_df = imp_df.sort_values("importance", ascending=True)

TOP_N = min(12, len(imp_df))
imp_top = imp_df.tail(TOP_N)

plt.figure(figsize=(7.2, 5.6), dpi=300)
plt.barh(
    imp_top["feature"], imp_top["importance"],
    color="#6B88C8", edgecolor="white", linewidth=0.7
)
plt.xlabel(x_label, fontsize=12)
plt.ylabel("")
plt.title("ANN (MLPRegressor) + StandardScaler — tuned α & lr_init", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.25)
plt.tight_layout()

# Save figure (timestamped, next to INPUT_PATH)
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(out_dir, f"importance_ann_tuned_{timestamp}.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print("Importance figure saved to:", out_png)

# ===== 5) Save importance tables & best params =====
out_xlsx = os.path.join(out_dir, f"ann_importance_tuned_{timestamp}.xlsx")
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
