import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create a sample time series data as a DataFrame
data = pd.read_csv('data/cleaned_type_feature_engineer.csv', parse_dates=True)
data = data.sort_index()

# Selecting multiple features (replace with your actual column names)
'''features = ['SF', 'Total_Rooms', 'Bds', 'Type_COND', 'Type_RENT', 'Type_RESI', 'Size Category_Large', 'Size Category_Medium', 'Size Category_Small', 'Stat_S', 'Stat_S-UL',
            'Agg_Median Days on Market', 'Agg_Months Supply of Inventory (Closed)', 'Agg_New Listings', 'Agg_Pending Sales', 'Agg_Homes for Sale']'''
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
print(features)
target = ['Log Price']

# Create lagged features
def create_lagged_features(df, lags):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    df.dropna(inplace=True)
    return df

# Parameters
n_lags = 500
# Create features and target
data = create_lagged_features(data, n_lags)
X = data[features].values
y = data[target].values

# Train-test split
# split_index = int(len(data) * (1 - test_size))
# X_train, X_test = X[:split_index], X[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]

n_forecast = 30
train_size = -1 - n_forecast
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Train the Random Forest model
model = RandomForestRegressor(n_estimators=3000, random_state=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Forecast future values
# Let's say we want to forecast the next 10 steps
last_observations = list(X[-1, :])

future_forecasts = []
for _ in range(n_forecast):
    forecast = model.predict(np.array(last_observations).reshape(1, -1))[0]
    future_forecasts.append(forecast)
    last_observations = last_observations[1:] + [forecast]

def plot_predictions(results):
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

results = pd.DataFrame({'Actual': y_test.flatten()[-1 - n_forecast:-1], 
                        'Predicted': future_forecasts})
plot_predictions(results)