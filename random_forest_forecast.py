import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create a sample time series data as a DataFrame
data = pd.read_csv('data/mixed_level/700_feature_engineer.csv')
features = ['SF', 'Agg_Homes for Sale']
target = ['List/Sell $']

X = data[features].values
Y = data[target].values

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a separate RandomForestRegressor for each target variable
models = []
for i in range(Y_train.shape[1]):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train[:, i])
    models.append(model)

# Make predictions
Y_pred = np.zeros(Y_test.shape)
for i, model in enumerate(models):
    Y_pred[:, i] = model.predict(X_test)

# Evaluate the model
for i in range(Y_test.shape[1]):
    mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
    print(f'MSE for target {i+1}: {mse}')

# Plotting predicted vs actual values
for i in range(Y_test.shape[1]):
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:, i], label='Actual')
    plt.plot(Y_pred[:, i], label='Predicted')
    plt.title(f'Target {i+1}: Actual vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.show()