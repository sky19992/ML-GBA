# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib

# === Font settings ===
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # keep SimHei for Chinese labels if needed
matplotlib.rcParams['axes.unicode_minus'] = False

# === Step 1: Read Excel data ===
df = pd.read_excel('final_Kd_database.xlsx')
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]

# === Step 2: Standardization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: RFECV feature selection ===
estimator = RandomForestRegressor(random_state=42)
rfecv = RFECV(
    estimator=estimator,
    step=1,  # could also use >1 or a fraction (0<step<1)
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1
)
rfecv.fit(X_scaled, y)

# === Reconstruct the “true” n_features path for the x-axis ===
if hasattr(rfecv, "cv_results_") and "n_features" in rfecv.cv_results_:
    n_features_path = np.asarray(rfecv.cv_results_["n_features"])
else:
    # Fallback for older versions: rebuild from start size and step
    start = X.shape[1]
    step = rfecv.step if isinstance(rfecv.step, int) else max(1, int(round(rfecv.step * start)))
    minf = getattr(rfecv, "min_features_to_select", 1)
    n_features_path = np.arange(start, minf - 1, -step)[::-1]

mean_score = np.asarray(rfecv.cv_results_['mean_test_score'])
std_score  = np.asarray(rfecv.cv_results_['std_test_score'])

best_idx = int(np.argmax(mean_score))
best_x   = int(n_features_path[best_idx])
best_y   = float(mean_score[best_idx])

# === Select the optimal features ===
support_mask  = rfecv.support_       # True/False mask
ranking       = rfecv.ranking_       # 1 = selected, >1 = not selected
selected_cols = X.columns[support_mask].tolist()

# === Optional: re-fit RF to get feature importances (for reference only) ===
estimator.fit(X_scaled, y)
importances = estimator.feature_importances_

# === Export to Excel (four sheets) ===
out_path = "final_Kd_database_selected.xlsx"

# 1) Keep only target + selected features
selected_df = pd.concat([y, X[selected_cols]], axis=1)
selected_df.to_excel(out_path, sheet_name="SelectedData", index=False)

# 2) Feature mask (boolean)
mask_df = pd.DataFrame({
    "Feature": X.columns,
    "Selected": support_mask
})
with pd.ExcelWriter(out_path, mode="a", engine="openpyxl") as writer:
    mask_df.to_excel(writer, sheet_name="FeatureMask", index=False)

# 3) Ranking + RF importance
rank_df = pd.DataFrame({
    "Feature": X.columns,
    "Ranking": ranking,
    "RF_Importance": importances
}).sort_values(by=["Ranking", "RF_Importance"], ascending=[True, False])
with pd.ExcelWriter(out_path, mode="a", engine="openpyxl") as writer:
    rank_df.to_excel(writer, sheet_name="Rankings", index=False)

# 4) CV curve data (true x-axis)
cv_curve_df = pd.DataFrame({
    "n_features": n_features_path,
    "mean_test_score": mean_score,
    "std_test_score": std_score
})
with pd.ExcelWriter(out_path, mode="a", engine="openpyxl") as writer:
    cv_curve_df.to_excel(writer, sheet_name="CV_Curve", index=False)

print(f"[OK] Exported optimal feature data to: {out_path}")
print(f"[INFO] Optimal number of features: {best_x}, Best mean R^2: {best_y:.4f}")
print("Selected features:", selected_cols)
