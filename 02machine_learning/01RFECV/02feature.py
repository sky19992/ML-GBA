# === Imports ===
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_squared_error

# === Step 1: Load data ===
DATA_PATH = 'final_Kd_database.xlsx'
df = pd.read_excel(DATA_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]

# === Step 2: Train/test split (split first; all fitting happens on training only) ===
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 3: Build pipeline: (optional) scaling -> RFECV(with RF) -> final RF ===
pipe = Pipeline([
    ("scaler", StandardScaler()),  # Not necessary for RF, but useful if you swap models later
    ("fs", RFECV(
        estimator=RandomForestRegressor(random_state=42, n_estimators=500),
        step=1,  # could also be >1 or a fraction (0 < step < 1)
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2',
        n_jobs=-1
    )),
    ("model", RandomForestRegressor(random_state=42, n_estimators=500))
])

# === Step 4: Fit on training set only (RFECV internally does CV and feature selection) ===
pipe.fit(X_tr, y_tr)

# === Step 5: Evaluate on the hold-out test set ===
y_pred = pipe.predict(X_te)
test_r2 = r2_score(y_te, y_pred)
test_rmse = mean_squared_error(y_te, y_pred, squared=False)
print(f"[METRIC] Test R^2: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")

# === Step 6: Inspect RFECV results and selected features ===
fs = pipe.named_steps["fs"]          # fitted RFECV object
final_rf = pipe.named_steps["model"] # fitted final RF (trained on selected features only)

# CV scores (compatible with older sklearn versions)
if hasattr(fs, "cv_results_"):
    mean_score = np.asarray(fs.cv_results_['mean_test_score'])
    std_score  = np.asarray(fs.cv_results_['std_test_score'])
    # n_features path (newer sklearn may provide it)
    if "n_features" in fs.cv_results_:
        n_features_path = np.asarray(fs.cv_results_["n_features"])
    else:
        start = X.shape[1]
        step = fs.step if isinstance(fs.step, int) else max(1, int(round(fs.step * start)))
        minf = getattr(fs, "min_features_to_select", 1)
        n_features_path = np.arange(start, minf - 1, -step)[::-1]
else:
    # Very old versions expose grid_scores_ only
    mean_score = np.asarray(fs.grid_scores_)
    std_score  = np.zeros_like(mean_score)
    start = X.shape[1]
    step = fs.step if isinstance(fs.step, int) else max(1, int(round(fs.step * start)))
    minf = getattr(fs, "min_features_to_select", 1)
    n_features_path = np.arange(start, minf - 1, -step)[::-1]

best_idx = int(np.argmax(mean_score))
best_x   = int(n_features_path[best_idx])
best_y   = float(mean_score[best_idx])
print(f"[INFO] Optimal number of features (CV): {best_x} | Best mean CV R^2: {best_y:.4f}")

# Selected feature mask and names (aligned with original X columns)
support_mask  = fs.support_
ranking       = fs.ranking_
selected_cols = X.columns[support_mask].tolist()
print("[INFO] Selected features:", selected_cols)

# === Step 7: Map feature importances from the final RF back to original columns ===
# final_rf.feature_importances_ corresponds only to the selected subset.
importances_selected = final_rf.feature_importances_
# Create a full-length vector (NaN for non-selected features)
rf_importances_full = np.full(shape=X.shape[1], fill_value=np.nan, dtype=float)
rf_importances_full[support_mask] = importances_selected

# === Step 8: Export results to a single Excel file (requires openpyxl) ===
out_path = "final_Kd_database_selected.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    # 1) Keep only target + selected features (all rows)
    selected_df = pd.concat([y, X[selected_cols]], axis=1)
    selected_df.to_excel(writer, sheet_name="SelectedData", index=False)

    # 2) Feature mask (boolean)
    mask_df = pd.DataFrame({
        "Feature": X.columns,
        "Selected": support_mask
    })
    mask_df.to_excel(writer, sheet_name="FeatureMask", index=False)

    # 3) Ranking + RF importance (from the final model on selected features)
    rank_df = pd.DataFrame({
        "Feature": X.columns,
        "Ranking": ranking,  # 1 = selected, >1 = not selected
        "RF_Importance_SelectedModel": rf_importances_full
    }).sort_values(by=["Ranking", "RF_Importance_SelectedModel"], ascending=[True, False], na_position="last")
    rank_df.to_excel(writer, sheet_name="Rankings", index=False)

    # 4) CV curve data (true x-axis path)
    cv_curve_df = pd.DataFrame({
        "n_features": n_features_path,
        "mean_test_score": mean_score,
        "std_test_score": std_score
    })
    cv_curve_df.to_excel(writer, sheet_name="CV_Curve", index=False)

    # 5) Hold-out test metrics
    metrics_df = pd.DataFrame([{"Test_R2": test_r2, "Test_RMSE": test_rmse}])
    metrics_df.to_excel(writer, sheet_name="TestMetrics", index=False)

print(f"[OK] Exported optimal feature data to: {out_path}")
