import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate synthetic data
np.random.seed(0)
data = np.random.rand(1000, 3)  # 1000 time steps, 3 features

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, :]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 10  # Length of the sequences
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, X_train.shape[2])))
model.add(Dense(X_train.shape[2]))  # Output layer should match the number of features

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_test_scaled = scaler.inverse_transform(y_test)
y_pred_scaled = scaler.inverse_transform(y_pred)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_scaled[:, 0], label='Actual Feature 1')
plt.plot(y_pred_scaled[:, 0], label='Predicted Feature 1')
plt.legend()
plt.show()
