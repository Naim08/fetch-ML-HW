import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col='# Date')

# Aggregate data to monthly
data_monthly = data['Receipt_Count'].resample('M').sum()

# Prepare the input features (X) and target variable (y)
data_monthly = data_monthly.reset_index()
data_monthly['Time_Index'] = np.arange(len(data_monthly))
X = np.column_stack((np.ones(len(data_monthly)), data_monthly['Time_Index']))  # Adding a column for the intercept
y = data_monthly['Receipt_Count'].values

# Calculate the coefficients using OLS formula
coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Making predictions for 2022 (12 months)
future_time_index = np.column_stack((np.ones(12), np.arange(len(data_monthly), len(data_monthly) + 12)))
future_predictions = future_time_index.dot(coefficients)

# Creating a DataFrame for the forecasted values
forecast_dates = pd.date_range(start=data_monthly['# Date'].iloc[-1] + pd.DateOffset(months=1), periods=12, freq='M')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Receipts': future_predictions})

# Plotting the actual data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(data_monthly['# Date'], y, label='Actual Monthly Receipts')
plt.plot(forecast_dates, future_predictions, label='Forecasted Receipts', color='red')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.title('Monthly Receipts Forecast for 2022')
plt.legend()
plt.show()

forecast_df
