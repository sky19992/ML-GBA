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

# === Fonts & plotting config ===
matplotlib.rcParams['font.family'] = ['Times New Roman']  # English-only plotting
matplotlib.rcParams['axes.unicode_minus'] = False

# === Step 1: Read Excel data ===
DATA_PATH = 'final_Kd_database.xlsx'   # <-- change if needed
df = pd.read_excel(DATA_PATH)

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
    step=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2',
    n_jobs=-1
)
rfecv.fit(X_scaled, y)

# Collect CV curve data
mean_score = rfecv.cv_results_['mean_test_score']
std_score = rfecv.cv_results_['std_test_score']
n_features = np.arange(1, len(mean_score) + 1)

best_idx = int(np.argmax(mean_score))
best_x = int(n_features[best_idx])
best_y = float(mean_score[best_idx])

# Report best results
print(f"Best number of features: {best_x}")
print(f"Best CV R^2: {best_y:.4f}")

# Save selected feature names
selected_mask = rfecv.support_
selected_features = X.columns[selected_mask]
pd.Series(selected_features, name='Selected_Features').to_csv('rfecv_selected_features.csv', index=False)
print("Selected feature names saved to: rfecv_selected_features.csv")

# === Step 4: Plot RFECV curve ===
plt.figure(figsize=(8, 6), dpi=300)
plt.errorbar(
    n_features, mean_score, yerr=std_score, fmt='-',
    color='brown', ecolor='lightcoral', elinewidth=0.8,
    label='Average score'
)

plt.plot(best_x, best_y, 'o', mfc='white', mec='brown')
plt.text(best_x + 2, best_y, f"{best_y:.2f}", fontsize=10)

plt.xlabel("Number of features", fontsize=14)
plt.ylabel("Average validation $R^2$", fontsize=14)
plt.xlim(0, len(n_features))
plt.ylim(0.50, 0.98)
plt.xticks(np.arange(0, len(n_features) + 1, 20))
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

# Save & show
plt.tight_layout()
plt.savefig("rfecv_curve.pdf", format="pdf", bbox_inches="tight")
plt.show()
