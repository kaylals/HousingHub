import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
# Example data: Replace with your time series data
# data = pd.read_csv('data/mixed_level/700_feature_engineer.csv').iloc[13145:]
data = pd.read_csv('data/mixed_level/700_feature_engineer.csv', index_col='Stat Date', parse_dates=True)
data = data.sort_index()

# Selecting multiple features (replace with your actual column names)
features = ['SF', 'Total_Rooms', 'Bds']
target = ['Log Price']

scaler_features = MinMaxScaler(feature_range=(0, 1))
features_data = pd.DataFrame(scaler_features.fit_transform(data[features]), 
                             columns=features, index=data.index)
target_data = data[target]

# Create sequences
def create_sequences(features, target, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features.iloc[i:(i+seq_length)].values
        y = target.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

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
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Make predictions
predictions = model.predict(X_test)

print("Sample predictions:")
for i in range(10):
    print(f"Actual: {y_test[i][0]:.4f}, Predicted: {predictions[i][0]:.4f}")


results_index = data.index[train_size+seq_length:]
results = pd.DataFrame({'Actual': y_test.flatten(), 
                        'Predicted': predictions.flatten()}, 
                       index=results_index)

def plot_predictions(results):
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Actual'], label='Actual Log Price')
    plt.plot(results.index, results['Predicted'], label='Predicted Log Price', linestyle='--')
    plt.title('Log Price Predictions vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Log Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_predictions(results)


sys.exit()

# Calculate error metrics
def calculate_metrics(predictions, actual):
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Calculate metrics
mae, mse, rmse = calculate_metrics(results['Predicted'], results['Actual'])

# Print metrics
print(f"Metrics for Log Price:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print()

# If you want to calculate percentage errors (which can be more interpretable for log-transformed data)
def calculate_percentage_errors(predictions, actual):
    percentage_errors = (predictions - actual) / actual * 100
    mape = np.mean(np.abs(percentage_errors))
    return mape

mape = calculate_percentage_errors(results['Predicted'], results['Actual'])
print(f"  MAPE: {mape:.2f}%")
