import os, re, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from nbeats_pytorch.model import NBeatsNet
import functools

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

# Save the modified DataFrame back to CSV if needed
data.to_csv('cleaned_type_feature_engineer.csv', index=False)

# Separate features and target
y = data['Log Price']
X = data.drop('Log Price', axis=1)

# Resample the data to a monthly frequency, taking the mean for each month
data_monthly = data.resample('M').mean()

# Separate features and target
y_monthly = data_monthly['Log Price']
X_monthly = data_monthly.drop('Log Price', axis=1)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled_monthly = pd.DataFrame(scaler_X.fit_transform(X_monthly), columns=X_monthly.columns, index=X_monthly.index)
y_scaled_monthly = pd.DataFrame(scaler_y.fit_transform(y_monthly.values.reshape(-1, 1)), columns=['Log Price'], index=y_monthly.index)

X_scaled_monthly = X_scaled_monthly.replace([np.inf, -np.inf], np.nan).dropna()
y_scaled_monthly = y_scaled_monthly.replace([np.inf, -np.inf], np.nan).dropna()

# Make sure X_scaled_monthly and y_scaled_monthly have the same index after dropping NaNs
common_index = X_scaled_monthly.index.intersection(y_scaled_monthly.index)
X_scaled_monthly = X_scaled_monthly.loc[common_index]
y_scaled_monthly = y_scaled_monthly.loc[common_index]

# Make sure X_scaled and y_scaled have the same index after dropping NaNs
common_index = X_scaled_monthly.index.intersection(y_scaled_monthly.index)
X_scaled = X_scaled_monthly.loc[common_index]
y_scaled = y_scaled_monthly.loc[common_index]

def create_dataset(X, y, input_steps, output_steps):
    X_data, y_data = [], []
    for i in range(len(X) - input_steps - output_steps + 1):
        X_data.append(X.iloc[i:(i + input_steps)].values.flatten())  # Flatten the input
        y_data.append(y.iloc[(i + input_steps):(i + input_steps + output_steps)].values.flatten())
    return np.array(X_data), np.array(y_data)

# Split into training and validation sets
train_size = int(len(X_scaled_monthly) * 0.8)

input_steps = 12  # Number of past months to use
output_steps = 6  # Number of future months to predict

X_train_monthly, X_val_monthly = X_scaled_monthly.iloc[:train_size], X_scaled_monthly.iloc[train_size:]
y_train_monthly, y_val_monthly = y_scaled_monthly.iloc[:train_size], y_scaled_monthly.iloc[train_size:]

X_train_monthly, y_train_monthly = create_dataset(X_train_monthly, y_train_monthly, input_steps, output_steps)
X_val_monthly, y_val_monthly = create_dataset(X_val_monthly, y_val_monthly, input_steps, output_steps)

train_dataset_monthly = TensorDataset(torch.Tensor(X_train_monthly), torch.Tensor(y_train_monthly))
val_dataset_monthly = TensorDataset(torch.Tensor(X_val_monthly), torch.Tensor(y_val_monthly))

train_loader_monthly = DataLoader(train_dataset_monthly, batch_size=16, shuffle=True)
val_loader_monthly = DataLoader(val_dataset_monthly, batch_size=16, shuffle=False)


class NBeatsNetWithDropout(NBeatsNet):
    def __init__(self, *args, **kwargs):
        super(NBeatsNetWithDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.2)  # Dropout with 20% probability

    def forward(self, x):
        backcast, forecast = super(NBeatsNetWithDropout, self).forward(x)
        forecast = self.dropout(forecast)  # Apply dropout to the forecast
        return backcast, forecast

def calculate_metrics(y_true, y_pred):
    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def load_model_and_losses():
    best_model = torch.load('best_nbeats_model.pth')
    model_monthly.load_state_dict(best_model)
    
    with open('losses.pkl', 'rb') as f:
        losses = pickle.load(f)
        return model_monthly, losses

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

@conditional_run(run_model=True)
def train_model():
    # Training loop
    train_losses = []
    val_losses = []
    epochs = 10
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model_monthly.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for x, y in train_loader_monthly:
            x = x.view(x.size(0), -1)  # Reshape input: (batch_size, input_length * num_features)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, forecast = model_monthly(x)

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
        train_losses.append(train_loss / len(train_loader_monthly))

        model_monthly.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in val_loader_monthly:
                x = x.view(x.size(0), -1)  # Reshape input
                x, y = x.to(device), y.to(device)
                _, forecast = model_monthly(x)
                val_loss += loss_fn(forecast, y).item()
                val_predictions.append(forecast.cpu().numpy())
                val_targets.append(y.cpu().numpy())
        
        val_losses.append(val_loss / len(val_loader_monthly))
        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)
        val_mae, val_mse, val_rmse = calculate_metrics(val_targets, val_predictions)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train - Loss: {train_loss/len(train_loader_monthly):.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}')
        print(f'Val - Loss: {val_loss/len(val_loader_monthly):.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model_monthly.state_dict()

    # Save the best model
    torch.save(best_model, 'best_nbeats_model.pth')
    
    # Save losses
    with open('losses.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

@conditional_plot(plot=True)
def plot_results():
    # Load the model and losses
    model, losses = load_model_and_losses()
    train_losses = losses['train_losses']
    val_losses = losses['val_losses']

    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for x, y in val_loader_monthly:
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Update the model initialization
model_monthly = NBeatsNetWithDropout(
    device=device,
    stack_types=(NBeatsNet.GENERIC_BLOCK,),
    forecast_length=output_steps,
    backcast_length=input_steps * X_monthly.shape[1],  # Multiply by number of features
    hidden_layer_units=16,
    nb_blocks_per_stack=3,
    thetas_dim=(4, 8),
    share_weights_in_stack=False
).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model_monthly.parameters(), lr=1e-5, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()


train_model()
plot_results()

sys.exit()

model_monthly.load_state_dict(torch.load('best_nbeats_model.pth'))
model_monthly.eval()

def get_monthly_prediction(input_date, bedrooms):
    # Convert input_date to datetime if it's not already
    input_date = pd.to_datetime(input_date)
    
    # Get the 12 months of data before the input date
    start_date = input_date - pd.DateOffset(months=12)
    input_data = X_scaled_monthly.loc[start_date:input_date].iloc[-12:]
    
    # If we don't have enough data, pad with zeros
    if len(input_data) < 12:
        pad_length = 12 - len(input_data)
        pad_data = pd.DataFrame(np.zeros((pad_length, X_scaled_monthly.shape[1])), columns=X_scaled_monthly.columns)
        input_data = pd.concat([pad_data, input_data])
    
    # Update the 'number_of_bedrooms' column with the input value
    input_data['Bds'] = bedrooms
    
    # Prepare the input for the model
    model_input = torch.Tensor(input_data.values.flatten()).unsqueeze(0).to(device)
    
    # Get the prediction
    with torch.no_grad():
        _, forecast = model_monthly(model_input)
    
    # Convert the forecast back to the original log scale
    forecast_log = scaler_y.inverse_transform(forecast.cpu().numpy().reshape(-1, 1))
    
    # Transform from log scale to original numeric scale
    forecast_numeric = np.exp(forecast_log)
    
    # Create a DataFrame with both forecasts
    forecast_dates = pd.date_range(start=input_date + pd.DateOffset(months=1), periods=6, freq='M')
    forecast_df = pd.DataFrame({
        'Predicted Log Price': forecast_log.flatten(),
        'Predicted Price': forecast_numeric.flatten()
    }, index=forecast_dates)
    
    return forecast_df

# Example usage
input_date = "2023-08-01"  # Replace with your desired date
bedrooms = 3  # Number of bedrooms
monthly_prediction = get_monthly_prediction(input_date, bedrooms)
fullprice_predictions = monthly_prediction['Predicted Price']
print(f"Prediction for 6 months starting from {input_date}:")
print(monthly_prediction)

