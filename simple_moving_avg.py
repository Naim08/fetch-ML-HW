import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your CSV data
data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col='# Date')

train_data = data[data.index < '2021-10-01']
test_data = data[data.index >= '2021-10-01']

# Choose a window size for the moving average
window_size = 30

# Calculate the moving average on the training set
train_data['SMA'] = train_data['Receipt_Count'].rolling(window=window_size).mean()

# Use the last SMA value from the training set as the forecast for the test set
sma_forecast = train_data['SMA'].iloc[-1]

# Create a Series of the SMA forecast with the same index as the test set
sma_predictions = pd.Series(sma_forecast, index=test_data.index)

# Calculate MSE between the actual values and SMA predictions
mse = mean_squared_error(test_data['Receipt_Count'], sma_predictions)
rsme = np.sqrt(mse)
print(f"Root Mean Squared Error of SMA: {rsme}")
print(f"Mean Squared Error of SMA: {mse}")
# Calculate the moving average
data['SMA'] = data['Receipt_Count'].rolling(window=window_size).mean()

# Plot the original data and the Simple Moving Average
plt.figure(figsize=(12, 6))
plt.plot(data['Receipt_Count'], label='Original Receipt Count')
plt.plot(data['SMA'], label=f'{window_size}-Day SMA', color='orange')
plt.title(f'Receipt Count and {window_size}-Day Simple Moving Average')
plt.xlabel('Date')
plt.ylabel('Receipt Count')
plt.legend()
plt.show()

# Forecasting the next period using the latest SMA value
# (Note: This is a simplistic forecast and may not be accurate for all scenarios)
latest_sma_forecast = data['SMA'].iloc[-1]
print(f"Forecast for the next period based on SMA: {latest_sma_forecast}")
