import os
import pandas as pd
import matplotlib
import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, send_file
from flask_cors import CORS
from flask import Flask, send_file, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
 
def plot_predictions(results, n_forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Predicted'], label='Predicted Log Price')
    plt.title('Log Price Predictions vs Actual')
    plt.xlabel(f'Last {n_forecast} days')
    plt.ylabel('Log Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("result/random_forest_forecast", 'acutals_vs_predictions.png'))
    plt.close()
 
# Create lagged features
def create_lagged_features(df, lags, target):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    df.dropna(inplace=True)
    return df
 
def prediction(start_date, end_date, bedrooms, bathrooms, property_type):
    # Create a sample time series data as a DataFrame
    data = pd.read_csv('data/cleaned_type_feature_engineer.csv', parse_dates=True)
    date = []
    for i in range(len(data)):
        x = datetime.datetime.strftime(datetime.datetime(data['Year'].loc[i], data['Month'].loc[i], data['Day'].loc[i]), "%Y-%m/%d")
        date.append(x)
    data["Date"] = date
    data = data.sort_values(by=["Date"])
    data = data.loc[(data['Bds'] == bedrooms) & data['Bths'] == bathrooms & (data['Date'] >= start_date) & (data['Date'] <= end_date)] 
    if property_type == "CONDO":
        data = data.loc[data['Type_COND'] == 1]
    elif property_type == "RESI":
        data = data.loc[data['Type_RESI'] == 1]

    # Selecting multiple features (replace with your actual column names)
    features = list(data.columns)
    features.remove('Log Price')
    features.remove('Year')
    features.remove('Month')
    features.remove('Day')
    features.remove('Agg_Average Price Per Square Foot')
    features.remove('Agg_Average Sales Price')
    features.remove('Agg_Median Price Per Square Foot')
    features.remove('Agg_Median Percent of Last Original Price')
    features.remove('Price_Per_SF')
    features.remove('Price_per_Bedroom')
    features.remove('Date')
    target = 'Log Price'

    
    # Parameters
    n_lags = 500
    # Create features and target
    data = create_lagged_features(data, n_lags, target)
    X = data[features].values
    y = data[target].values

    n_forecast = 60
    train_size = -1 - n_forecast
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]


    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=30, random_state=1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    results = pd.DataFrame({'Predicted': y_pred})
    plot_predictions(results, n_forecast)

def rf_api():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400

    start_date = data.get('startDate')
    end_date = data.get('endDate')
    bedrooms = data.get('bedrooms')
    bathrooms = data.get('bathrooms')
    property_type = data.get('type')

    prediction(start_date, end_date, bedrooms, bathrooms, property_type)
    
    return send_file("../../result/random_forest_forecast/acutals_vs_predictions.png"), 200