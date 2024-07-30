import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example data: Replace with your time series data
data = pd.read_csv('data/mixed_level/700_feature_engineer.csv').iloc[13145:]

# Selecting multiple features (replace with your actual column names)
features = ['SF', 'Total_Rooms', 'Bds']
target = ['Log Price']

# Normalize the data
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

# Separate and scale input and target data
features_data = scaler_features.fit_transform(data[features].values)
target_data = scaler_target.fit_transform(data[target].values)

# Create sequences
def create_sequences(features, target, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length - 1):
        x = features[i:(i+seq_length)]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 50  # Sequence length
X, y = create_sequences(features_data, target_data, seq_length)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=10)

# Evaluate the model
model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler_target.inverse_transform(predictions)  # Inverse transform to original scale
actual = scaler_target.inverse_transform(y_test)

# For further analysis, you can compare the predictions with the actual values
def plot_predictions(predictions, actual, feature_index=0, feature_name='Feature'):
    plt.figure(figsize=(12, 6))
    plt.plot(actual[:, feature_index], color='blue', label=f'Actual {feature_name}')
    plt.plot(predictions[:, feature_index], color='red', linestyle='--', label=f'Predicted {feature_name}')
    plt.title(f'{feature_name} Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel(feature_name)
    plt.legend()
    plt.show()

for i, feature in enumerate(target):
    plot_predictions(predictions, actual, feature_index=i, feature_name=feature)

# Calculate error metrics
def calculate_metrics(predictions, actual):
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Calculate metrics for each feature
metrics = {}
for i, feature in enumerate(target):
    mae, mse, rmse = calculate_metrics(predictions[:, i], actual[:, i])
    metrics[feature] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# Print metrics
for feature, values in metrics.items():
    print(f"Metrics for {feature}:")
    print(f"  MAE: {values['MAE']:.4f}")
    print(f"  MSE: {values['MSE']:.4f}")
    print(f"  RMSE: {values['RMSE']:.4f}")
    print()