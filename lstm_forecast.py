import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Convert to pandas DataFrame for easier manipulation
data = pd.read_csv('data/mixed_level/700_feature_engineer.csv')

df = data[['SF', 'Agg_Homes for Sale', 'List/Sell $']]

# Normalize the dataset
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Convert to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# Define function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][-1]  # Assuming target is the last column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Parameters
SEQ_LENGTH = 10

# Create sequences
X, y = create_sequences(scaled_df.values, SEQ_LENGTH)

# Split into train and test sets
SPLIT = int(0.8 * len(X))
X_train, X_test = X[:SPLIT], X[SPLIT:]
y_train, y_test = y[:SPLIT], y[SPLIT:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions to original scale
predictions = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], predictions], axis=1))[:, -1]

# Inverse transform actual values to original scale
y_test_orig = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], y_test.reshape(-1, 1)], axis=1))[:, -1]

# Plot results
plt.figure(figsize=(14, 5))
plt.plot(y_test_orig, color='blue', label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.legend()
plt.show()