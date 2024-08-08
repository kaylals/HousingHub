import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import torch
from datetime import datetime

# Load the data
input_path = "data/aggregated_700.csv"
output_folder = "result/prophet"

data = pd.read_csv(input_path, index_col='Date', parse_dates=True)
data = data.sort_index()

# Create a new directory to save the figures
output_dir = 'result/prophet'
os.makedirs(output_dir, exist_ok=True)

# Define a function to analyze a time series variable and save the figures
def analyze_time_series(variable):
    # Trend Analysis
    plt.figure(figsize=(12, 6))
    data[variable].plot(title=f'Trend Analysis of {variable}')
    plt.xlabel('Date')
    plt.ylabel(variable)
    plt.savefig(os.path.join(output_dir, f'{variable}_trend.png'))
    plt.close()

    # Handle missing values (either drop or fill)
    series = data[variable].dropna()  # You can use .fillna(method='ffill') to forward-fill instead

    if series.isnull().any():
        print(f"Warning: {variable} contains missing values after dropping nulls.")

    # Seasonal Decomposition
    decomposition = seasonal_decompose(series, model='additive', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle(f'Seasonal Decomposition of {variable}', y=1.02)
    plt.savefig(os.path.join(output_dir, f'{variable}_decomposition.png'))
    plt.close()

    # Prophet Forecasting
    df_prophet = series.reset_index().rename(columns={'Date': 'ds', variable: 'y'})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)
    
    # Make future dataframe for predictions
    future = model.make_future_dataframe(periods=365)  # Adjust periods for forecast length
    forecast = model.predict(future)

    # Ensure 'ds' columns are datetime for merging
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Save the model using torch
    torch.save(model, f'{variable}_prophet_model.pth')

# Function to load the model and make predictions
def predict_inventory(variable, start_year_month, end_year_month):
    # Load the whole model
    model = torch.load(f'{variable}_prophet_model.pth')
    
    # Create a future dataframe for predictions
    start_date = datetime.strptime(start_year_month, '%Y-%m')
    end_date = datetime.strptime(end_year_month, '%Y-%m')
    future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    future = pd.DataFrame(future_dates, columns=['ds'])
    
    # Make predictions
    forecast = model.predict(future)
    
    # Extract predictions as a list of lists with date and predicted value
    predictions = forecast[['ds', 'yhat']].values.tolist()
    
    return predictions

# Example usage
analyze_time_series('Median Days on Market')
predictions = predict_inventory('Median Days on Market', '2024-01', '2024-12')
print(predictions)
