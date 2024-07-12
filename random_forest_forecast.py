import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create a sample time series data as a DataFrame
dates = pd.read_csv('combined_df_normalized.csv')['Date']
time_series_values = pd.read_csv('combined_df_normalized.csv')['Average Sales Price']
time_series_data = pd.DataFrame({'date': dates, 'value': time_series_values})

# Create lagged features
def create_lagged_features(df, lags):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df.dropna(inplace=True)
    return df

# Parameters
n_lags = 12
test_size = 0.2

# Create features and target
data = create_lagged_features(time_series_data, n_lags)
X = data.drop(['date', 'value'], axis=1)
y = data['value']

# Train-test split
split_index = int(len(data) * (1 - test_size))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Forecast future values
# Let's say we want to forecast the next 10 steps
n_forecast = 10
last_observations = list(X.iloc[-1, :].values)

future_forecasts = []
for _ in range(n_forecast):
    forecast = model.predict(np.array(last_observations).reshape(1, -1))[0]
    future_forecasts.append(forecast)
    last_observations = last_observations[1:] + [forecast]

print(f'Future forecasts: {future_forecasts}')