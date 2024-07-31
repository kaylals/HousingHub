# pip install torch nbeats-pytorch

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import pickle



# Load your data
input_path = "data/mixed_level/700_feature_engineer.csv"
output_folder = "result/n_beats"


data = pd.read_csv('data/mixed_level/700_feature_engineer.csv', index_col='Stat Date', parse_dates=True)
data = data.sort_index()

# Scale the data
scaler = MinMaxScaler()
# data['Log Price'] = scaler.fit_transform(data[['Log Price']])

# Split into training and validation sets
train_size = int(len(data) * 0.8)
train, val = data.iloc[:train_size], data.iloc[train_size:]


def create_dataset(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:(i + input_steps), 0])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), 0])
    return np.array(X), np.array(y)

input_steps = 30  # Number of past observations to use
output_steps = 7  # Number of future steps to predict

X_train, y_train = create_dataset(train.values, input_steps, output_steps)
X_val, y_val = create_dataset(val.values, input_steps, output_steps)


import torch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


from nbeats_pytorch.model import NBeatsNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = NBeatsNet(
    device=device,
    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
    forecast_length=output_steps,
    backcast_length=input_steps,
    hidden_layer_units=128
).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Training loop
train_losses = []
val_losses = []
epochs = 50
best_val_loss = float('inf')
best_model = None

# for epoch in range(epochs):
#     model.train()
#     train_loss = 0
#     train_predictions = []
#     train_targets = []
    
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         _, forecast = model(x)
#         loss = loss_fn(forecast, y)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         train_predictions.append(forecast.detach().cpu().numpy())
#         train_targets.append(y.detach().cpu().numpy())
    
#     train_losses.append(train_loss / len(train_loader))
#     train_predictions = np.concatenate(train_predictions)
#     train_targets = np.concatenate(train_targets)
#     train_mae, train_mse, train_rmse = calculate_metrics(train_targets, train_predictions)
    
#     model.eval()
#     val_loss = 0
#     val_predictions = []
#     val_targets = []
    
#     with torch.no_grad():
#         for x, y in val_loader:
#             x, y = x.to(device), y.to(device)
#             _, forecast = model(x)
#             val_loss += loss_fn(forecast, y).item()
#             val_predictions.append(forecast.cpu().numpy())
#             val_targets.append(y.cpu().numpy())
    
#     val_losses.append(val_loss / len(val_loader))
#     val_predictions = np.concatenate(val_predictions)
#     val_targets = np.concatenate(val_targets)
#     val_mae, val_mse, val_rmse = calculate_metrics(val_targets, val_predictions)
    
#     print(f'Epoch {epoch+1}/{epochs}')
#     print(f'Train - Loss: {train_loss/len(train_loader):.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}')
#     print(f'Val - Loss: {val_loss/len(val_loader):.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')
    
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_model = model.state_dict()

# # After training
# with open('losses.pkl', 'wb') as f:
#     pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

# Save the best model
# torch.save(best_model, 'best_nbeats_model.pth')

best_model = torch.load('best_nbeats_model.pth', weights_only=True)
model.load_state_dict(best_model)

# Load the losses
with open('losses.pkl', 'rb') as f:
    losses = pickle.load(f)
    train_losses = losses['train_losses']
    val_losses = losses['val_losses']


model.eval()
final_predictions = []
final_targets = []
 
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, forecast = model(x)
        final_predictions.append(forecast.cpu().numpy())
        final_targets.append(y.cpu().numpy())

final_predictions = np.concatenate(final_predictions)
final_targets = np.concatenate(final_targets)

import matplotlib.pyplot as plt

# Plotting the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.savefig(os.path.join(output_folder, 'loss_curves.png'))
plt.show()



import matplotlib.pyplot as plt

# Ensure data index is datetime
data.index = pd.to_datetime(data.index)

# Get the last 60 days of data
last_60_days = data.last('60D')

# Calculate the train_size and get the corresponding date
train_size = int(len(data) * 0.8)
train_end_date = data.index[train_size]

# Create mask for the last 60 days of predictions
pred_dates = data.index[train_size:train_size+len(final_predictions)]
last_60_pred_mask = pred_dates >= last_60_days.index[0]

# Actual vs Predicted plot
plt.figure(figsize=(15, 8))

# Plot actual values
plt.plot(last_60_days.index, last_60_days.iloc[:, 0], label='Actual', color='blue')

# Plot predicted values
plt.plot(pred_dates[last_60_pred_mask], 
         final_predictions[last_60_pred_mask, 0], 
         label='Predicted', 
         color='red')

plt.legend()
plt.title('Actual vs Predicted Time Series (Last 60 Days)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'actual_vs_predicted_last_60_days.png'))
plt.close()

# Residuals plot
plt.figure(figsize=(15, 8))

# Calculate residuals for the available predictions
residuals = data.iloc[train_size:train_size+len(final_predictions), 0].values - final_predictions[:, 0]

# Plot residuals for the last 60 days
plt.plot(pred_dates[last_60_pred_mask], residuals[last_60_pred_mask], label='Residuals')

plt.title('Residuals of Predictions (Last 60 Days)')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'residuals_last_60_days.png'))
plt.close()

# Time Series Forecast plot
plt.figure(figsize=(15, 10))

# Plot actual data
plt.plot(last_60_days.index, last_60_days.iloc[:, 0], label='Actual Data', color='blue')

# Plot predictions
for i in range(final_predictions.shape[1]):
    plt.plot(pred_dates[last_60_pred_mask], 
             final_predictions[last_60_pred_mask, i], 
             label=f'Forecast t+{i+1}', 
             alpha=0.7)

plt.legend()
plt.title('Time Series Forecast (Last 60 Days)')
plt.xlabel('Date')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'time_series_forecast_last_60_days.png'))
plt.close()