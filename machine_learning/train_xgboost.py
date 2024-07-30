import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset with the correct relative path
df = pd.read_csv('../data/mixed_level/700_feature_engineer.csv', low_memory=False)


# Convert all other columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Define target variable and features
target = 'Log Price'  
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Split the data into 85% for training (including cross-validation) and 15% for testing
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Further split the training data for cross-validation (85% of 85% is ~72%)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100, 200],
    'colsample_bytree': [0.3, 0.7]
}

# Initialize the XGBoost model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# Set up Grid Search with cross-validation on the training set
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_xg_reg = grid_search.best_estimator_

# Evaluate the best model on the validation set
y_val_pred = best_xg_reg.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2_val = r2_score(y_val, y_val_pred)

print("Best parameters found: ", best_params)
print("Validation RMSE: %f" % (rmse_val))
print("Validation R²: %f" % (r2_val))

# Evaluate the best model on the test set
y_test_pred = best_xg_reg.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("Test RMSE: %f" % (rmse_test))
print("Test R²: %f" % (r2_test))



# # Plot feature importance
# xgb.plot_importance(best_xg_reg)
# plt.show()

# Function to plot learning curves
# def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     plt.xlabel("Training examples")
#     plt.ylabel("RMSE")

#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')

#     train_scores_mean = np.mean(-train_scores, axis=1)
#     train_scores_std = np.std(-train_scores, axis=1)
#     test_scores_mean = np.mean(-test_scores, axis=1)
#     test_scores_std = np.std(-test_scores, axis=1)

#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt

# # Plot learning curves for the best model
# plot_learning_curve(best_xg_reg, "Learning Curves (XGBoost)", X_train_full, y_train_full, cv=5)
# plt.show()



# residuals = y_test - y_test_pred
# plt.scatter(y_test_pred, residuals)
# plt.xlabel("Predicted Values")
# plt.ylabel("Residuals")
# plt.title("Residual Plot")
# plt.axhline(y=0, color='r', linestyle='-')
# plt.show()
