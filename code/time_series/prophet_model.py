# pip install prophet


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet  # Import Prophet

# Load the data
input_path = "data/mixed_level/700_feature_engineer.csv"
output_folder = "result/prophet"

data = pd.read_csv('data/mixed_level/700_feature_engineer.csv', index_col='Stat Date', parse_dates=True)
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
    df_prophet = series.reset_index().rename(columns={'Stat Date': 'ds', variable: 'y'})
    model = Prophet()
    model.fit(df_prophet)
    
    # Make future dataframe for predictions
    future = model.make_future_dataframe(periods=60)  # Adjust periods for forecast length
    forecast = model.predict(future)


    last_60_days = data.last('60D')
    pred_dates = forecast['ds']
    last_60_pred_mask = pred_dates >= last_60_days.index[0]

    # Plot actual vs predicted
    plt.figure(figsize=(15, 8))
    plt.plot(last_60_days.index, last_60_days[variable], label='Actual', color='blue')
    plt.plot(pred_dates[last_60_pred_mask], 
            forecast['yhat'][last_60_pred_mask], 
            label='Predicted', 
            color='red')
    plt.legend()
    plt.title('Actual vs Predicted Time Series (Last 60 Days)')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'prophet_actual_vs_predicted_last_60_days.png'))
    plt.close()

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    plt.suptitle(f'Prophet Forecast Components for {variable}', y=1.02)
    plt.savefig(os.path.join(output_dir, f'{variable}_prophet_components.png'))
    plt.close()

# Get the list of variables (all columns except 'Date')
variables = ['Log Price']

# Analyze each variable
for var in variables:
    print(f'Analyzing {var}...')
    analyze_time_series(var)
