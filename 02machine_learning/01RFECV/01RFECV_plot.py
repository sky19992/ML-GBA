import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load data
DATA_PATH = 'final_Kd_database.xlsx'
df = pd.read_excel(DATA_PATH)
target_col = df.columns[0]
X = df.drop(columns=[target_col])
y = df[target_col]

# Train/test split (split first; all subsequent fits occur on the training set only)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=3047)

# Pipeline: (optional) scaling -> RFECV (with RF) -> final RF
pipe = Pipeline([
    ("scaler", StandardScaler()),  # Not necessary for RF, but kept for easy model swapping later
    ("fs", RFECV(
        estimator=RandomForestRegressor(random_state=3047, n_estimators=500),
        step=1,
        cv=KFold(n_splits=5, shuffle=True, random_state=3047),
        scoring="r2",
        n_jobs=-1
    )),
    ("model", RandomForestRegressor(random_state=3047, n_estimators=500))
])

# Fit only on the training set (RFECV internally performs CV and feature selection)
pipe.fit(X_tr, y_tr)

# Evaluate on the test set
y_pred = pipe.predict(X_te)
test_r2 = r2_score(y_te, y_pred)
test_rmse = mean_squared_error(y_te, y_pred, squared=False)
print(f"Test R^2: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")

# Extract RFECV step to inspect CV curve and selected features
fs = pipe.named_steps["fs"]

# Compatibility with older sklearn versions
if hasattr(fs, "cv_results_"):
    mean_score = fs.cv_results_['mean_test_score']
    std_score  = fs.cv_results_['std_test_score']
else:
    mean_score = fs.grid_scores_
    std_score  = np.zeros_like(mean_score)

n_features = np.arange(1, len(mean_score) + 1)
best_idx = int(np.argmax(mean_score))
best_x = int(n_features[best_idx])
best_y = float(mean_score[best_idx])
print(f"Best number of features (CV): {best_x}, Best CV R^2: {best_y:.4f}")

# Save selected feature names (note: fs.support_ aligns with the original X column order)
selected_mask = fs.support_
selected_features = X.columns[selected_mask]
pd.Series(selected_features, name='Selected_Features').to_csv('rfecv_selected_features.csv', index=False)
print("Selected feature names saved to: rfecv_selected_features.csv")

# Plot RFECV curve
plt.figure(figsize=(8, 6), dpi=300)
plt.errorbar(n_features, mean_score, yerr=std_score, fmt='-', color='brown',
             ecolor='lightcoral', elinewidth=0.8, label='Average score')
plt.plot(best_x, best_y, 'o', mfc='white', mec='brown')
plt.text(best_x + 2, best_y, f"{best_y:.2f}", fontsize=10)

plt.xlabel("Number of features", fontsize=14)
plt.ylabel("Average validation $R^2$", fontsize=14)
plt.xlim(0, len(n_features))
plt.ylim(min(0.50, mean_score.min()-0.02), min(0.99, mean_score.max()+0.02))
step = max(1, len(n_features)//20)
plt.xticks(np.arange(0, len(n_features) + 1, step))
plt.yticks(np.arange(0.50, 0.99, 0.05))
plt.legend(loc='upper right', frameon=False)

# Inset zoom plot
axins = inset_axes(plt.gca(), width="35%", height="25%", loc='lower left', borderpad=2)
zoom_slice = slice(max(0, best_idx - 5), best_idx + 6)
axins.plot(n_features[zoom_slice], mean_score[zoom_slice], color='brown', linewidth=1.5)
axins.set_xticks(n_features[zoom_slice][::2])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_color('gray')

plt.tight_layout()
plt.savefig("rfecv_curve.pdf", format="pdf", bbox_inches="tight")
plt.show()
