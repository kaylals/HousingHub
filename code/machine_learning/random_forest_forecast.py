import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from flask import Flask
from flask import send_file

app = Flask(__name__)

def plot_predictions(results, n_forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(results['Predicted'], label='Predicted Price')
    plt.title('Price Predictions')
    plt.xlabel(f'Last {n_forecast} days')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("result/random_forest_forecast", 'acutals_vs_predictions.png'))

# Create lagged features
def create_lagged_features(df, lags, target):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    df.dropna(inplace=True)
    return df

def prediction(start_date, range_dates, bedrooms, bathrooms, property_type):
    # Create a sample time series data as a DataFrame
    data = pd.read_csv('data/cleaned_type_feature_engineer.csv', parse_dates=True)
    date = []
    for i in range(len(data)):
        x = datetime.datetime.strftime(datetime.datetime(data['Year'].loc[i], data['Month'].loc[i], data['Day'].loc[i]), "%Y-%m/%d")
        date.append(x)
    data["Date"] = date
    data = data.sort_values(by=["Date"])
    data = data.loc[(data['Bds'] == bedrooms) & (data['Bths'] == bathrooms) & (data['Date'] >= start_date)] 
    if property_type == "CONDO":
        data = data.loc[data['Type_COND'] == 1]
    elif property_type == "RENT":
        data = data.loc[data['Type_RENT'] == 1]
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
    target = ['Log Price']

    
    # Parameters
    n_lags = 500
    # Create features and target
    data = create_lagged_features(data, n_lags, target)
    X = data[features].values
    y = data[target].values

    n_forecast = range_dates * 30
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

    results = pd.DataFrame({'Predicted': np.exp(y_pred)})
    plot_predictions(results, n_forecast)

@app.get("/random-forest-forecast")
def api():
    start_date = '2001-01-01'
    range_months = 2
    bedrooms = 1
    bathrooms = 1
    property_type = "CONDO"
    prediction(start_date, range_months, bedrooms, bathrooms, property_type)
    return send_file("../../result/random_forest_forecast/acutals_vs_predictions.png")

if __name__ == '__main__':
   app.run()