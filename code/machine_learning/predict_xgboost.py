import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load the trained model
model_filename = 'best_xgboost_model.pkl'
best_xg_reg = joblib.load(model_filename)

# Load the dataset to get mean values for missing features
df = pd.read_csv('../../data/mixed_level/700_feature_engineer.csv', low_memory=False)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Define the pattern to match columns that start with "Type" and end with numbers
import re
pattern = re.compile(r'^Type_\d+$')

# Filter out columns that match the pattern
columns_to_drop = [col for col in df.columns if pattern.match(col)]
df = df.drop(columns=columns_to_drop)

# Remove additional specific high VIF features identified
features_to_remove = ['Price_per_Bedroom', 'Size Category_Medium', 'Total_Rooms', 'Type_COND']
df_reduced = df.drop(columns=features_to_remove)

# Define target variable and features
target = 'Log Price'
features = df_reduced.columns[df_reduced.columns != target]

# Define the function to get predictions
def get_prediction(start_date, range_months, bedrooms, bathrooms):
    # Convert start_date to datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate date range at monthly intervals
    future_dates = [start_date + relativedelta(months=i) for i in range(range_months)]
    
    # Create a DataFrame for future dates with the provided features
    future_data = pd.DataFrame({
        'Bds': bedrooms,
        'Bths': bathrooms,
        'Type_RESI': 1,  # Assuming the property is residential
        'Stat_RESI': 1,  # Assuming the property status is residential
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
    
    return aggregated_data

# Example usage
start_date = '2024-08-01'
range_months = 24  # Predict for the next 24 months
bedrooms = [3] * range_months  # Assume 3 bedrooms for each month
bathrooms = [2] * range_months  # Assume 2 bathrooms for each month

predictions = get_prediction(start_date, range_months, bedrooms, bathrooms)
print(predictions)

