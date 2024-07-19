import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

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

    # ARIMA Modeling
    model = ARIMA(data[variable], order=(5, 1, 0))  # ARIMA(p, d, q) order
    arima_result = model.fit()
    summary_text = arima_result.summary().as_text()
    
    # Save the ARIMA summary to a text file
    summary_file_path = os.path.join(output_dir, f'{variable}_arima_summary.txt')
    with open(summary_file_path, 'w') as f:
        f.write(summary_text)

    # Plot the fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(data[variable], label='Original')
    plt.plot(arima_result.fittedvalues, color='red', label='Fitted')
    plt.title(f'ARIMA Model Fitting for {variable}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{variable}_arima.png'))
    plt.close()

# Get the list of variables (all columns except 'Date')
variables = data.columns

# Analyze each variable
for var in variables:
    print(f'Analyzing {var}...')
    analyze_time_series(var)