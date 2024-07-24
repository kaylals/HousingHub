# pip install prophet


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet  # Import Prophet

# Load the data
file_path = './combined_df_normalized.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Create a new directory to save the figures
output_dir = './result/time_series_analysis_results'
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

    # Seasonal Decomposition
    decomposition = seasonal_decompose(data[variable], model='additive', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle(f'Seasonal Decomposition of {variable}', y=1.02)
    plt.savefig(os.path.join(output_dir, f'{variable}_decomposition.png'))
    plt.close()

    # Prophet Forecasting
    df_prophet = data.reset_index().rename(columns={'Date': 'ds', variable: 'y'})
    model = Prophet()
    model.fit(df_prophet)
    
    # Make future dataframe for predictions
    future = model.make_future_dataframe(periods=365)  # Adjust periods for forecast length
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f'Prophet Forecast for {variable}')
    plt.savefig(os.path.join(output_dir, f'{variable}_prophet_forecast.png'))
    plt.close()

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    plt.suptitle(f'Prophet Forecast Components for {variable}', y=1.02)
    plt.savefig(os.path.join(output_dir, f'{variable}_prophet_components.png'))
    plt.close()

# Get the list of variables (all columns except 'Date')
variables = data.columns

# Analyze each variable
for var in variables:
    print(f'Analyzing {var}...')
    analyze_time_series(var)
