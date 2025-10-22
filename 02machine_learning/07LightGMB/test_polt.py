# === SHAP-based Feature Importance — LightGBM (robust, with CV R² & exports) ===
# pip install lightgbm shap scikit-optimize openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from lightgbm import LGBMRegressor
import shap
from datetime import datetime
from textwrap import shorten


INPUT_PATH = r"final_Kd_database_selected.xlsx"
TEST_SIZE = 0.2
RANDOM_STATE = 3047
N_CV = 5
N_BAYES_ITER = 48
TOP_N = 12
USE_FULL_DATA_FOR_IMPORTANCE = False   # 论文稳妥：False -> 仅用训练集（或其子样本）计算 SHAP
MAX_SHAP_SAMPLES = 1200               # 为提速的上限子样本数
FIG_W, FIG_H, DPI = 7.2, 5.6, 300

# ===== 1) Load Excel =====
df = pd.read_excel(INPUT_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]


# ===== 2) Train / Test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ===== 3) LightGBM + BayesSearchCV=====
lgbm = LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    n_estimators=500,
    verbose=-1
)

param_space = {
    "n_estimators":      Integer(200, 700),
    "num_leaves":        Integer(15, 256),
    "learning_rate":     Real(0.01, 0.2, prior="log-uniform"),
    "feature_fraction":  Real(0.5, 1.0),   # colsample_bytree
    "bagging_fraction":  Real(0.5, 1.0),   # subsample
    "bagging_freq":      Integer(1, 7),
    "min_child_samples": Integer(5, 100),
    "min_split_gain":    Real(0.0, 1.0),
    "reg_alpha":         Real(0.0, 1.0),
    "reg_lambda":        Real(0.0, 2.0),
}

cv_inner = KFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_STATE)
bayes = BayesSearchCV(
    estimator=lgbm,
    search_spaces=param_space,
    n_iter=N_BAYES_ITER,
    cv=cv_inner,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)
bayes.fit(X_train, y_train)
best_model = bayes.best_estimator_
best_params = bayes.best_params_
print("Best Parameters:", best_params)

# ===== 4) Train/Test metrics =====
y_train_pred = best_model.predict(X_train)
y_test_pred  = best_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
r2_train   = r2_score(y_train, y_train_pred)
r2_test    = r2_score(y_test,  y_test_pred)
mae_train  = mean_absolute_error(y_train, y_train_pred)
mae_test   = mean_absolute_error(y_test,  y_test_pred)
mdae_train = np.median(np.abs(y_train - y_train_pred))
mdae_test  = np.median(np.abs(y_test  - y_test_pred))

print(f"Train R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
print(f"Test  R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

# ===== 5) 5-fold CV R²=====
cv = KFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_STATE)
cv_r2 = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
print("5-fold CV R² (train):", np.round(cv_r2, 4),
      "Mean =", round(cv_r2.mean(), 4), "Std =", round(cv_r2.std(), 4))


# ---------------- 6) Export results to Excel ----------------
out_dir = os.path.dirname(INPUT_PATH) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(out_dir, f"shap_importance_lgbm_{timestamp}.png")
plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
plt.show()
print("Importance figure saved to:", out_png)

out_xlsx = os.path.join(out_dir, f"lgbm_importance_{timestamp}.xlsx")
with pd.ExcelWriter(out_xlsx) as writer:
    pd.DataFrame([best_params]).to_excel(writer, sheet_name="BestParams", index=False)
    # 训练/测试指标
    pd.DataFrame([{
        "R2_train": r2_train, "RMSE_train": rmse_train, "MAE_train": mae_train, "MdAE_train": mdae_train,
        "R2_test":  r2_test,  "RMSE_test":  rmse_test,  "MAE_test":  mae_test,  "MdAE_test":  mdae_test
    }]).to_excel(writer, sheet_name="Metrics", index=False)
    # 5 折 CV R²
    pd.DataFrame({"fold": np.arange(1, len(cv_r2)+1), "R2": cv_r2}).to_excel(
        writer, sheet_name="CV5_R2_train_folds", index=False
    )
    pd.DataFrame([{"CV5_R2_mean": cv_r2.mean(), "CV5_R2_std": cv_r2.std()}]).to_excel(
        writer, sheet_name="CV5_R2_train_summary", index=False
    )

print("Tables saved to:", out_xlsx)
