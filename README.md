01change_sdf_descriptor
1-1)
1-2)





2-1) all Kd values were log-transformed to reduce variability;
2-2) outliers in log Kd were identified and removed using the interquartile range (IQR) method; 
2-3) Spearman correlation analysis was conducted to assess multicollinearity among environmental factors and molecular descriptors; 
2-4) highly correlated variables were eliminated, with the remaining features retained as predictors and log Kd as the response variable. 
02machine_learning
To ensure a sample-to-feature ratio (SFR) greater than 10 and determine the optimal feature set, 
Recursive Feature Elimination with Cross-Validation (RFECV) based on Random Forest was applied for feature reduction. 
For comparison, a multiple linear regression (MLR) model was first established as a baseline, and its predictive performance was compared with Random Forest (RF), 
Extreme Gradient Boosting (XGBoost), Support Vector Regression (SVR), Artificial Neural Network (ANN), and Light Gradient Boosting Machine (LightGBM). 
The dataset was split into training (80%) and testing (20%) subsets, with five-fold cross-validation and Bayesian optimization applied to the training set for hyperparameter tuning, ensuring an optimal balance between bias and variance. 
Model performance was comprehensively evaluated using the coefficient of determination (R²) and root mean square error (RMSE). 
