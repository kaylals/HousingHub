import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import time
import matplotlib.pyplot as plt

# Load the dataset with the correct relative path
df = pd.read_csv('../data/mixed_level/700_feature_engineer.csv', low_memory=False)

# Drop the 'MLS' column
df.drop('MLS', axis=1, inplace=True)

# Convert 'Stat Date' to datetime
df['Stat Date'] = pd.to_datetime(df['Stat Date'], errors='coerce')

# Extract year, month, and day from 'Stat Date'
df['Stat Year'] = df['Stat Date'].dt.year
df['Stat Month'] = df['Stat Date'].dt.month
df['Stat Day'] = df['Stat Date'].dt.day

# Drop the original 'Stat Date' column
df.drop('Stat Date', axis=1, inplace=True)

# Convert all other columns to numeric, coerce errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with the mean (or other appropriate method)
df.fillna(df.mean(), inplace=True)

# Define target variable and features
target = 'List/Sell $'  
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'colsample_bytree': [0.3, 0.7]
}


# Initialize the XGBoost model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# Set up Grid Search
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_xg_reg = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_best = best_xg_reg.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print("Best parameters found: ", best_params)
print("Best RMSE: %f" % (rmse_best))
print("Best R²: %f" % (r2_best))

# # Plot feature importance
# xgb.plot_importance(best_xg_reg)
# plt.show()

