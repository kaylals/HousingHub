import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from nbeats_pytorch.model import NBeatsNet
from flask import Flask
from flask import send_file

app = Flask(__name__)

# Load your data
input_path = "data/mixed_level/700_feature_engineer.csv"
output_folder = "result/n_beats"

data = pd.read_csv(input_path, index_col='Stat Date', parse_dates=True)
data = data.sort_index()

# Define the pattern to match columns that start with "Type" and end with numbers
pattern = re.compile(r'^Type_\d+$')

# Filter out columns that match the pattern
columns_to_drop = [col for col in data.columns if pattern.match(col)]
data = data.drop(columns=columns_to_drop)

# # Save the modified DataFrame back to CSV if needed
# data.to_csv('data/cleaned_type_feature_engineer.csv', index=False)

# Separate features and target
y = data['Log Price']
X = data.drop('Log Price', axis=1)


conditions = [
    data['Type_COND'] == 1,
    data['Type_RESI'] == 1
]

choices = ['CONDO', 'RESI']

data['type'] = np.select(conditions, choices)

# Scale the data
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
y_scaled = pd.DataFrame(scaler.fit_transform(y.values.reshape(-1, 1)), columns=['Log Price'], index=y.index)

X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).dropna()
y_scaled = y_scaled.replace([np.inf, -np.inf], np.nan).dropna()

# Make sure X_scaled and y_scaled have the same index after dropping NaNs
common_index = X_scaled.index.intersection(y_scaled.index)
X_scaled = X_scaled.loc[common_index]
y_scaled = y_scaled.loc[common_index]

def create_dataset(X, y, input_steps, output_steps):
    X_data, y_data = [], []
    for i in range(len(X) - input_steps - output_steps + 1):
        X_data.append(X.iloc[i:(i + input_steps)].values.flatten())  # Flatten the input
        y_data.append(y.iloc[(i + input_steps):(i + input_steps + output_steps)].values.flatten())
    return np.array(X_data), np.array(y_data)

# Split into training and validation sets
train_size = int(len(data) * 0.8)
input_steps = 90  # Number of past observations to use
output_steps = 30  # Number of future steps to predict

X_train, X_val = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
y_train, y_val = y_scaled.iloc[:train_size], y_scaled.iloc[train_size:]

X_train, y_train = create_dataset(X_train, y_train, input_steps, output_steps)
X_val, y_val = create_dataset(X_val, y_val, input_steps, output_steps)

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class NBeatsNetWithDropout(NBeatsNet):
    def __init__(self, *args, **kwargs):
        super(NBeatsNetWithDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.2)  # Dropout with 20% probability

    def forward(self, x):
        backcast, forecast = super(NBeatsNetWithDropout, self).forward(x)
        forecast = self.dropout(forecast)  # Apply dropout to the forecast
        return backcast, forecast

# Load the saved model
model = NBeatsNetWithDropout(
    device=device,
    stack_types=(NBeatsNet.GENERIC_BLOCK,),
    forecast_length=30,  # Assuming you're predicting 7 days ahead
    backcast_length=90 * X.shape[1],  # Adjust based on your input size
    hidden_layer_units=32,
    nb_blocks_per_stack=4,
    thetas_dim=(4, 8),
    share_weights_in_stack=False
).to(device)


# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

def calculate_metrics(y_true, y_pred):
    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

import functools

def conditional_run(run_model=True):
    def decorator_run_model(func):
        @functools.wraps(func)
        def wrapper_run_model(*args, **kwargs):
            if run_model:
                return func(*args, **kwargs)
            else:
                print(f"Skipping {func.__name__}")
                return None
        return wrapper_run_model
    return decorator_run_model

def conditional_plot(plot=True):
    def decorator_plot(func):
        @functools.wraps(func)
        def wrapper_plot(*args, **kwargs):
            if plot:
                return func(*args, **kwargs)
            else:
                print(f"Skipping {func.__name__}")
                return None
        return wrapper_plot
    return decorator_plot

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=10, min_delta=0.001)

@conditional_run(run_model=False)
def train_model():
    # Training loop
    train_losses = []
    val_losses = []
    epochs = 30
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for x, y in train_loader:
            x = x.view(x.size(0), -1)  # Reshape input: (batch_size, input_length * num_features)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, forecast = model(x)

            # Check for NaN values
            if torch.isnan(forecast).any() or torch.isnan(y).any():
                print(f"NaN detected in epoch {epoch+1}")
                continue

            loss = loss_fn(forecast, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_predictions.append(forecast.detach().cpu().numpy())
            train_targets.append(y.detach().cpu().numpy())
        
        train_predictions = np.concatenate(train_predictions)
        train_targets = np.concatenate(train_targets)

        # Check for NaN values before calculating metrics
        if np.isnan(train_predictions).any() or np.isnan(train_targets).any():
            print(f"NaN detected in predictions or targets in epoch {epoch+1}")
            continue

        train_mae, train_mse, train_rmse = calculate_metrics(train_targets, train_predictions)
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.view(x.size(0), -1)  # Reshape input
                x, y = x.to(device), y.to(device)
                _, forecast = model(x)
                val_loss += loss_fn(forecast, y).item()
                val_predictions.append(forecast.cpu().numpy())
                val_targets.append(y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)
        val_mae, val_mse, val_rmse = calculate_metrics(val_targets, val_predictions)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train - Loss: {train_loss/len(train_loader):.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}')
        print(f'Val - Loss: {val_loss/len(val_loader):.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        # Call early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save the best model
    torch.save(best_model, 'best_nbeats_model_by_day.pth')
    
    # Save losses
    with open('losses.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

def load_model_and_losses():
    best_model = torch.load('best_nbeats_model_by_day.pth')
    model.load_state_dict(best_model)
    
    with open('losses.pkl', 'rb') as f:
        losses = pickle.load(f)
        return model, losses

@conditional_plot(plot=False)
def plot_results():
    # Load the model and losses
    model, losses = load_model_and_losses()
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

train_model()
plot_results()



model.load_state_dict(torch.load('best_nbeats_model_by_day.pth'))
model.eval()




# Define the prediction function
def get_prediction(start_date, range_dates, bedrooms, bathrooms, type):
    # Ensure range_dates is within the model's capability
    if range_dates > 30:
        raise ValueError("range_dates cannot exceed 30 days")
    
    # Convert start_date to datetime if it's not already
    start_date = pd.to_datetime(start_date)
    
    # Calculate end_date based on the range of days
    end_date = start_date + pd.Timedelta(days=range_dates-1)
    
    # Filter the data based on user input for bedrooms and bathrooms
    filtered_data = X_scaled[(X_scaled['Bds'] == bedrooms) & 
                             (X_scaled['Bths'] == bathrooms) &
                             X_scaled['type'].str.contains(type)]
    
    # Select data for the input window (90 days before start_date)
    input_start = start_date - pd.Timedelta(days=90)
    input_data = filtered_data.loc[input_start:start_date]
    
    # If we don't have enough data, pad with zeros
    if len(input_data) < 90:
        pad_length = 90 - len(input_data)
        pad_data = pd.DataFrame(np.zeros((pad_length, filtered_data.shape[1])), columns=filtered_data.columns)
        input_data = pd.concat([pad_data, input_data])
    
    # Prepare the input for the model
    model_input = torch.Tensor(input_data.values.flatten()).unsqueeze(0).to(device)
    
    # Get the prediction
    with torch.no_grad():
        _, forecast = model(model_input)
    
    # Convert the forecast back to the original scale
    forecast_log = scaler.inverse_transform(forecast.cpu().numpy().reshape(-1, 1))
    
    # Transform from log scale to original numeric scale
    forecast_numeric = np.exp(forecast_log)
    
    # Create a DataFrame with both forecasts
    forecast_dates = pd.date_range(start=start_date, periods=30)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates.strftime('%Y-%m-%d'),
        'Predicted Log Price': np.round(forecast_log.flatten(), 2),
        'Predicted Price': forecast_numeric.flatten().astype(int)
    })
    
    # Select only the requested range of dates
    forecast_df = forecast_df.loc[forecast_df['Date'].between(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]

    plot_predictions(forecast_df, range_dates)
    # Return the lists of dates, log prices, and prices
    return forecast_df['Date'].tolist(), forecast_df['Predicted Log Price'].tolist(), forecast_df['Predicted Price'].tolist()

def plot_predictions(results, days):
    plt.figure(figsize=(12, 6))
    plt.plot(results['Predicted Price'], label='Predicted Price')
    plt.title('Price Predictions')
    plt.xlabel(f'Last {days} days')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("result/n_beats", f'{days}_predicted.png'))


@app.get("/random-forest-forecast")
def api():
    start_date = '2024-01-01'
    range_days = 15
    bedrooms = 1
    bathrooms = 1
    property_type = "CONDO"
    get_prediction(start_date, range_days, bedrooms, bathrooms, property_type)
    return send_file("../../result/random_forest_forecast/acutals_vs_predictions.png")

if __name__ == '__main__':
   app.run()