import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset with the correct relative path
df = pd.read_csv('../data/mixed_level/700_feature_engineer.csv', low_memory=False)

# Drop the 'MLS' and 'List/Sell $' columns as they are not required for training
df.drop(['MLS', 'List/Sell $'], axis=1, inplace=True)

# Convert 'Stat Date' to datetime
df['Stat Date'] = pd.to_datetime(df['Stat Date'], errors='coerce')

# Extract year, month, and day from 'Stat Date'
df['Stat Year'] = df['Stat Date'].dt.year
df['Stat Month'] = df['Stat Date'].dt.month
df['Stat Day'] = df['Stat Date'].dt.day

# Drop the original 'Stat Date' column
df.drop('Stat Date', axis=1, inplace=True)

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

# Plot feature importance
xgb.plot_importance(best_xg_reg)
plt.show()
