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
input_path = "'data/cleaned_type_feature_engineer.csv"
output_folder = "result/n_beats"

data = pd.read_csv(input_path)


# Separate features and target
y = data['Log Price']
X = data.drop('Log Price', axis=1)


conditions = [
    data['Type_COND'] == 1,
    data['Type_RESI'] == 1
]

choices = [0, 1]

data['type'] = np.select(conditions, choices, default=0)

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


model.load_state_dict(torch.load('best_nbeats_model_by_day.pth'))
model.eval()


# Define the prediction function
def get_prediction(start_date, range_dates=15, bedrooms=2, bathrooms=2, type='CONDO'):
    # Ensure range_dates is within the model's capability
    if range_dates > 30:
        raise ValueError("range_dates cannot exceed 30 days")
    
    # Convert start_date to datetime if it's not already
    start_date = pd.to_datetime(start_date)
    
    # Calculate end_date based on the range of days
    end_date = start_date + pd.Timedelta(days=range_dates-1)
    type_mapping = {"condo": 0, "resi": 1}
    mapped_type = type_mapping.get(type)

    # Filter the data based on user input for bedrooms and bathrooms
    filtered_data = X_scaled[(X_scaled['Bds'] == bedrooms) & 
                             (X_scaled['Bths'] == bathrooms) &
                             X_scaled['type'] == mapped_type]
    
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

from flask import Flask, send_file, request, jsonify

counter = 1
@app.post("/n-beats-forecast")
def api():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400
    
    start_date = data.get('startDate')
    # end_date = data.get('endDate')
    range_days = int(data.get('range'))   # Predict for the next 24 months
    bedrooms = int(data.get('bedrooms'))   # Assume 3 bedrooms for each month
    bathrooms = int(data.get('bathrooms'))   # Assume 2 bathrooms for each month
    property_type = data.get('type')  # User-selected property type (either 'CONDO' or 'RESI')

    result, status_code = get_prediction(start_date, range_days, bedrooms, bathrooms, property_type), 200
    
    if status_code == 200:
        # Use an absolute path
        output_dir = os.path.join(os.path.dirname(__file__), "result", "nbeats")
        os.makedirs(output_dir, exist_ok=True)
        
        # Find the next available file number
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("predictions_") and f.endswith(".png")]
        existing_numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        next_number = max(existing_numbers, default=0) + 1
        
        # Generate the new filename
        file_name = f"predictions_nbeats_{next_number}.png"
        file_path = os.path.join(output_dir, file_name)
        
        return send_file(file_path, mimetype='image/png')
    
    return jsonify(result), status_code

if __name__ == '__main__':
   app.run()