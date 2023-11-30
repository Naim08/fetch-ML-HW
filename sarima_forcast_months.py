import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data_daily.csv'
data = pd.read_csv(file_path)

# Rename the '# Date' column to 'Date' and convert to datetime
data.rename(columns={'# Date': 'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index and aggregate the data on a monthly basis for 2021
data.set_index('Date', inplace=True)
monthly_data_2021 = data[data.index.year == 2021].resample('M').sum()
monthly_data_2021.reset_index(inplace=True)

# Fit the SARIMA model
p, d, q = 1, 1, 1  # ARIMA parameters
P, D, Q, S = 1, 1, 1, 5  # Seasonal parameters
model = SARIMAX(monthly_data_2021['Receipt_Count'], order=(p, d, q), seasonal_order=(P, D, Q, S))
results = model.fit()

# Generate forecasts for the next 12 months (2022)
forecast = results.get_forecast(steps=12)
predicted_receipts = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({
    'Month': pd.date_range(start='2022-01-01', periods=12, freq='M'),
    'Predicted Receipts': predicted_receipts,
    'Lower CI': confidence_intervals.iloc[:, 0],
    'Upper CI': confidence_intervals.iloc[:, 1]
})

# Display the forecast DataFrame
plt.figure(figsize=(12, 6))
plt.plot(monthly_data_2021['Date'], monthly_data_2021['Receipt_Count'], label='Actual Receipts')
plt.plot(forecast_df['Month'], forecast_df['Predicted Receipts'], label='Forecasted Receipts', color='red')
plt.fill_between(forecast_df['Month'], forecast_df['Lower CI'], forecast_df['Upper CI'], alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.title('Monthly Receipts Forecast for 2022')
plt.legend()
plt.show()

forecast_df.to_csv('data/sarima_forecasted_monthly_values.csv', index=False)
