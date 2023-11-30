import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


# Load your data
data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col='# Date')

# Split the data into training and test sets
train_data = data[data.index < '2021-10-01']
test_data = data[data.index >= '2021-10-01']

# Define SARIMA Parameters
p, d, q = 1, 1, 1  # Non-seasonal parameters
P, D, Q, S = 1, 1, 1, 30 # Seasonal parameters (example: yearly seasonality with S=12)

# Fit the SARIMA model
model = SARIMAX(train_data['Receipt_Count'], order=(p, d, q), seasonal_order=(P, D, Q, S))
results = model.fit()

# Forecast future values
forecast_steps = len(test_data)
forecast_output = results.get_forecast(steps=forecast_steps)
forecasted_values = forecast_output.predicted_mean

# Evaluate the model
mse = mean_squared_error(test_data['Receipt_Count'], forecasted_values)
print(f"Mean Squared Error: {mse}")

# Convert the forecast output into a DataFrame for export
forecast_df = pd.DataFrame(forecasted_values)
forecast_df.columns = ['Forecasted_Value']
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

forecast_df.to_csv('sarima_forecasted_values.csv')

# Optional: Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data['Receipt_Count'], label='Training Data')
plt.plot(test_data['Receipt_Count'], label='Test Data')
plt.plot(forecasted_values, label='Forecast', color='red')
plt.legend()
plt.show()
