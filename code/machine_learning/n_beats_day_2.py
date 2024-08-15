import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nbeats_pytorch.model import NBeatsNet
from flask import Flask, send_file, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import sys


logging.basicConfig(level=logging.INFO)

# app = Flask(__name__)
# CORS(app)

# Load your data
input_path = "cleaned_type_feature_engineer.csv"
output_folder = "result/n_beats"

data = pd.read_csv(input_path)

data['Price'] = np.exp(data['Log Price'])
# data['Type'] = data['Type_RENT'] + data['Type_COND']

# columns_to_drop = ['Type_RENT', 'Type_COND', 'Type_RESI', 'Log Price'] 
columns_to_drop = ['Log Price'] 
data = data.drop(columns=columns_to_drop)

# Separate features and target
y = data['Price']
X = data.drop('Price', axis=1)

# Scale the data
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).dropna()
y_scaled = y.replace([np.inf, -np.inf], np.nan).dropna()


# Make sure X_scaled and y_scaled have the same index after dropping NaNs
common_index = X_scaled.index.intersection(y_scaled.index)
X_scaled = X_scaled.loc[common_index]
y_scaled = y_scaled.loc[common_index]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NBeatsNetWithDropout(NBeatsNet):
    def __init__(self, *args, **kwargs):
        super(NBeatsNetWithDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.2)  # Dropout with 20% probability
        # self.sigmoid = nn.Sigmoid()
        self.scaling_factor = 450

    def forward(self, x):
        backcast, forecast = super(NBeatsNetWithDropout, self).forward(x)
        forecast = self.dropout(forecast)  # Apply dropout to the forecast
        # forecast = self.sigmoid(forecast)
        forecast = forecast * self.scaling_factor
        return backcast, forecast

# Load the saved model
model = NBeatsNetWithDropout(
    device=device,
    stack_types=(NBeatsNet.GENERIC_BLOCK,),
    forecast_length=30,  # Assuming you're predicting 30 days ahead
    backcast_length=90 * X.shape[1],  # Adjust based on your input size
    hidden_layer_units=32,
    nb_blocks_per_stack=4,
    thetas_dim=(4, 8),
    share_weights_in_stack=False
).to(device)


model.load_state_dict(torch.load('best_nbeats_model_by_day.pth'))
model.eval()
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

def plot_predictions(results, file_path):
    plt.figure(figsize=(12, 6))
    results['Date'] = pd.to_datetime(results['Date'])
    plt.plot(results['Date'], results['Predicted Price'], label='Predicted Price')
    plt.title('Predicted Property Prices Over Time')
    plt.xlabel('Next days')
    plt.ylabel('Price (USD)')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${int(x):,}'))
    # Formatting the x-axis
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    min_price = results['Predicted Price'].min()
    max_price = results['Predicted Price'].max()
    padding = (max_price - min_price) * 0.05  # Add 5% padding on either side
    plt.ylim(min_price - padding, max_price + padding)

    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def get_prediction(start_date, range_dates=15, bedrooms=2, bathrooms=2, property_type='CONDO'):
    # Ensure range_dates is within the model's capability
    if range_dates > 30:
        raise ValueError("range_dates cannot exceed 30 days")
    
    # Convert start_date to datetime if it's not already
    start_date = pd.to_datetime(start_date)
    
    # Calculate end_date based on the range of days
    end_date = start_date + pd.Timedelta(days=range_dates-1)
    type_mapping = {"CONDO": 0, "RESI": 1}
    mapped_type = type_mapping.get(property_type)

    # Filter the data based on user input for bedrooms and bathrooms
    filtered_data = X_scaled[(X_scaled['Bds'] == bedrooms) & 
                             (X_scaled['Bths'] == bathrooms)]
    
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
    
    # Create a DataFrame with both forecasts
    forecast_dates = pd.date_range(start=start_date, periods=30)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates.strftime('%Y-%m-%d'),
        'Predicted Price': forecast.flatten().int().numpy()
    })
    
    # Select only the requested range of dates
    forecast_df = forecast_df.loc[forecast_df['Date'].between(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]

    return forecast_df

# Example usage
# start_date = "2023-08-01"  # Replace with your desired start date
# range_dates = 15           # Replace with the desired range in days (up to 30)
# bedrooms = 3
# bathrooms = 2
# forecast_df = get_prediction(start_date, range_dates, bedrooms, bathrooms, 'CONDO')
# plot_predictions(forecast_df, 'forecast.png')
# sys.exit()

# @app.post("/n-beats-forecast")
def nb_api():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400
    
    start_date = data.get('startDate')
    range_days = data.get('range')
    bedrooms = data.get('bedrooms')
    bathrooms = data.get('bathrooms')
    property_type = data.get('type')

    output_dir = os.path.join(os.path.dirname(__file__), "result", "nbeats")
    os.makedirs(output_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("predictions_nbeats_") and f.endswith(".png")]
    existing_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
    next_number = max(existing_numbers, default=0) + 1
    
    file_name = f"predictions_nbeats_{next_number}.png"
    file_path = os.path.join(output_dir, file_name)

    forecast_df = get_prediction(start_date, range_days, bedrooms, bathrooms, property_type)
    plot_predictions(forecast_df, file_path)
    
    if os.path.isfile(file_path):
        return send_from_directory(output_dir, file_name, mimetype='image/png')
    else:
        return jsonify({'error': 'Image not produced'}), 500

# if __name__ == '__main__':
#    app.run()