import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
from io import BytesIO
import matplotlib.ticker as ticker

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load the trained model
script_dir = os.path.dirname(__file__)
model_filename = os.path.join(script_dir, 'best_xgboost_model.pkl')
best_xg_reg = joblib.load(model_filename)

# Load the dataset to get mean values for missing features
data_filename = os.path.join(script_dir, 'data/mixed_level/700_feature_engineer.csv')
df = pd.read_csv('data/mixed_level/700_feature_engineer.csv', low_memory=False)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Define the pattern to match columns that start with "Type" and end with numbers
import re
pattern = re.compile(r'^Type_\d+$')

# Filter out columns that match the pattern
columns_to_drop = [col for col in df.columns if pattern.match(col)]
df = df.drop(columns=columns_to_drop)

# Remove additional specific high VIF features identified and drop Type_RENT
features_to_remove = ['Price_per_Bedroom', 'Size Category_Medium', 'Total_Rooms', 'Type_RENT']
df_reduced = df.drop(columns=features_to_remove)

# Define target variable and features
target = 'Log Price'
features = df_reduced.columns[df_reduced.columns != target]

def plot_predictions(results):
     # Ensure that the 'Year-Month' is treated as a categorical variable
    results['Year-Month'] = results['Year-Month'].astype(str)
    
    # Convert 'Predicted_Price' to a float for plotting
    # results['Predicted_Price'] = results['Predicted_Price'].replace('[\$,]', '', regex=True).astype(float)
    results['Predicted_Price'] = results['Predicted_Price'].replace(r'[\$,]', '', regex=True).astype(float)

    plt.figure(figsize=(12, 6))
    plt.plot(results['Year-Month'], results['Predicted_Price'], marker='o', label='Predicted Price')
    plt.title('Predicted Property Prices Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)

    # Format the y-axis to show the price in dollar format
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${int(x):,}'))

    plt.tight_layout()

    # Save the plot
    save_dir = os.path.join(os.path.dirname(__file__), "result", "xgboost")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'predictions.png'))
    plt.close()

# Define the function to get predictions
def get_prediction(start_date, range_months, bedrooms, bathrooms, property_type):
    # Convert start_date to datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate date range at monthly intervals
    future_dates = [start_date + relativedelta(months=i) for i in range(range_months)]
    
    # Initialize the property type features
    type_cond = 0
    type_resi = 0
    
    if property_type == 'CONDO':
        type_cond = 1
    if property_type == 'RESI':
        type_resi = 1

    # Create a DataFrame for future dates with the provided features
    future_data = pd.DataFrame({
        'Bds': bedrooms,
        'Bths': bathrooms,
        'Type_RESI': type_resi,  # User-selected property type
        'Stat_RESI': type_resi,  # Assuming the property status matches the selected type
        'Type_COND': type_cond,  # User-selected property type
        'Stat_S': 0,  # Assuming the property is not sold
        'Stat_S-UL': 0,  # Assuming the property is not under contract,
        'Month': [date.month for date in future_dates],  # Add Month as a feature to capture seasonality
        'Year': [date.year for date in future_dates]     # Add Year for additional temporal context
    }, index=future_dates)
    
    # Add other required features with their mean values, reducing variance
    mean_values = df_reduced.mean()
    std_devs = df_reduced.std()
    for col in features:
        if col not in future_data.columns:
            if col in ['Year', 'Month', 'Day']:
                future_data[col] = [date.year if col == 'Year' else date.month if col == 'Month' else date.day for date in future_dates]
            else:
                mean_value = mean_values[col]
                std_dev = std_devs[col] / 5  # Reduce the variance to 1/5th
                future_data[col] = mean_value + np.random.normal(0, std_dev, size=range_months)
    
    # Ensure all expected columns are in the future_data, setting missing ones to zero
    for col in features:
        if col not in future_data.columns:
            future_data[col] = 0
    
    # Drop the target column if it exists in the DataFrame
    if target in future_data.columns:
        future_data = future_data.drop(columns=[target])
    
    # Predict the future log prices
    future_log_prices = best_xg_reg.predict(future_data[features])
    
    # Convert log prices to actual prices
    future_prices = np.exp(future_log_prices)
    
    # Add predictions to the DataFrame
    future_data['Predicted_Price'] = future_prices
    
    # Reset the index to include the date as a column
    future_data.reset_index(inplace=True)
    future_data.rename(columns={'index': 'Date'}, inplace=True)
    
    # Convert predicted prices to numeric type for aggregation
    future_data['Predicted_Price'] = future_data['Predicted_Price'].astype(float)
    
    # Aggregate predictions by month and year
    future_data['Year-Month'] = future_data['Date'].dt.to_period('M')
    aggregated_data = future_data.groupby('Year-Month').agg({'Predicted_Price': 'mean'}).reset_index()
    
    # Format the aggregated predicted prices for display
    aggregated_data['Predicted_Price'] = aggregated_data['Predicted_Price'].apply(lambda x: f"${x:,.2f}")
    plot_predictions(aggregated_data)
    return aggregated_data

def xg_api():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400
    
    start_date = data.get('startDate')
    # end_date = data.get('endDate')
    range_months = int(data.get('range') / 30)   # Predict for the next 24 months
    bedrooms = int(data.get('bedrooms')) * range_months  # Assume 3 bedrooms for each month
    bathrooms = int(data.get('bathrooms')) * range_months  # Assume 2 bathrooms for each month
    property_type = data.get('type')  # User-selected property type (either 'CONDO' or 'RESI')

    result, status_code = get_prediction(start_date, range_months, bedrooms, bathrooms, property_type), 200
    
    if status_code == 200:
        # Use an absolute path
        file_path = os.path.join(os.path.dirname(__file__), "result", "xgboost", "predictions.png")
        return send_file(file_path, mimetype='image/png')
    
    return jsonify(result), status_code