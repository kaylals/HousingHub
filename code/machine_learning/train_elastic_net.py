import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset with the correct relative path
df = pd.read_csv('./data/mixed_level/700_feature_engineer.csv', low_memory=False)

# Drop the 'MLS' column as it is just an identifier
df.drop('MLS', axis=1, inplace=True)

# Convert 'Stat Date' to datetime
df['Stat Date'] = pd.to_datetime(df['Stat Date'], errors='coerce')

# Extract year, month, and day from 'Stat Date'
df['Stat Year'] = df['Stat Date'].dt.year
df['Stat Month'] = df['Stat Date'].dt.month
df['Stat Day'] = df['Stat Date'].dt.day

# Drop the original 'Stat Date' column and 'Agg_Date' column
df.drop(['Stat Date', 'Agg_Date'], axis=1, inplace=True)

# Convert all other columns to numeric, coerce errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values by imputing the mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Define target variable and features, including 'Log Price'
target = 'List/Sell $'
features = [col for col in df_imputed.columns if col != target]

X = df_imputed[features]
y = df_imputed[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the ElasticNet model
elastic_net = ElasticNet()

# Define the parameter grid
param_grid = {
    'alpha': [0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 0.9]
}

# Set up Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_elastic_net = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_best = best_elastic_net.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print("Best parameters found: ", best_params)
print("Best RMSE: %f" % (rmse_best))
print("Best R²: %f" % (r2_best))

# Feature importance (coefficients)
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': best_elastic_net.coef_
})

print(coefficients.sort_values(by='Coefficient', ascending=False))
