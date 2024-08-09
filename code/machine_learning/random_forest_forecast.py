import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from flask import Flask

app = Flask(__name__)

def plot_predictions(results, n_forecast=60):
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Actual'], label='Actual Log Price')
    plt.plot(results.index, results['Predicted'], label='Predicted Log Price', linestyle='--')
    plt.title('Log Price Predictions vs Actual')
    plt.xlabel(f'Last {n_forecast} days')
    plt.ylabel('Log Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Create lagged features
def create_lagged_features(df, lags, target):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    df.dropna(inplace=True)
    return df

def prediction(start_date, range_dates, bedrooms=1, bathrooms=1):
    # Create a sample time series data as a DataFrame
    data = pd.read_csv('data/cleaned_type_feature_engineer.csv', parse_dates=True)
    data = data.sort_index()
    data = data.loc[(data['Bds'] == bedrooms) & data['Bths'] == bathrooms] 

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
    target = ['Log Price']

    
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

    results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred})

    return results

results = prediction(0, 0)

@app.get("/model1")
def api():
    return results.to_json()

if __name__ == '__main__':
   app.run()