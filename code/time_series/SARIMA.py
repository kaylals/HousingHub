import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys

# Load the data
input_path = 'data/aggregated_700_normalized.csv'
data = pd.read_csv(input_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Create a new directory to save the figures
output_dir = 'data/result/time_series_analysis_results'
os.makedirs(output_dir, exist_ok=True)



from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    adf_test = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {adf_test[0]}')
    print(f'p-value: {adf_test[1]}')
    for key, value in adf_test[4].items():
        print(f'Critical Values {key}: {value}')

# Get the list of variables (all columns except 'Date')
variables = data.columns


for var in variables:
    print(f'Analyzing {var}...')
    test_stationarity(data[var])


sys.exit()






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

    # SARIMA Modeling
    model = SARIMAX(data[variable], 
                    order=(5, 1, 0),  # ARIMA(p, d, q) order
                    seasonal_order=(1, 1, 1, 12))  # Seasonal order (P, D, Q, S)
    sarima_result = model.fit()
    summary_text = sarima_result.summary().as_text()
    
    # Save the ARIMA summary to a text file
    summary_input_path = os.path.join(output_dir, f'{variable}_sarima_summary.txt')
    with open(summary_input_path, 'w') as f:
        f.write(summary_text)

    # Plot the fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(data[variable], label='Original')
    plt.plot(sarima_result.fittedvalues, color='red', label='Fitted')
    plt.title(f'SARIMA Model Fitting for {variable}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{variable}_sarima.png'))
    plt.close()



# Analyze each variable
for var in variables:
    print(f'Analyzing {var}...')
    analyze_time_series(var)