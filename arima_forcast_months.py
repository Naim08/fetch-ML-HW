

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

def fit_ar_model(data, p):
    """
    Fit an autoregressive (AR) model of order p.
    """
    # Create shifted versions of the data for the AR model
    X = np.column_stack([data.shift(i) for i in range(1, p + 1)])
    y = data.values[p:]

    # Ensure that X and y have the same length
    min_length = min(len(X), len(y))
    X = X[:min_length]
    y = y[:min_length]

    # Remove NaNs caused by shifting
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    # Add a column for the intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Solve the linear regression
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    return coefficients


def forecast_arima(data, ar_coefficients, ma_coefficients, p, q, steps):
    """
    Forecast future values using an ARIMA model (AR and MA components).
    """
    forecast = []
    data_extended = list(data.values)
    errors = [0] * q  # Initialize errors for the MA component

    for t in range(steps):
        # AR Component
        lag_values = data_extended[-p:]
        forecast_value = ar_coefficients[0] + np.sum(ar_coefficients[1:] * lag_values)

        # MA Component
        forecast_value += np.sum(ma_coefficients * errors[-q:])

        # Update errors
        if t < len(data):  # Check if actual data exists for calculating error
            error = data[t] - forecast_value
        else:
            error = 0  # No actual data for future steps, assume no error
        errors.append(error)

        forecast.append(forecast_value)
        data_extended.append(forecast_value)

    return forecast

# Load and preprocess your data
data = pd.read_csv('data_daily.csv', parse_dates=['Date'], index_col='Date')

# Aggregate data to monthly
data_monthly = data['Receipt_Count'].resample('M').sum()

# Split the data into training and test sets (adjust dates accordingly)
train_data = data_monthly[data_monthly.index < '2021-10-01']
test_data = data_monthly[data_monthly.index >= '2021-10-01']

# Assuming p and d values are already determined
p = 1  # Order of the AR part
d = 0  # Differencing order
best_aic = np.inf
best_q = 0

# Find the best 'q' value based on AIC
for q in range(10):  # Testing q values from 0 to 9
    try:
        model = sm.tsa.ARIMA(train_data, order=(p, d, q))
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_q = q
    except:  # Handle cases where the model fails to converge
        continue

print(f"Best q value: {best_q} with AIC: {best_aic}")

# Fit the ARIMA model with the best 'q' value found
model = sm.tsa.ARIMA(train_data, order=(p, d, best_q))
results = model.fit()

forecast_steps = len(test_data)
forecast_output = results.forecast(steps=forecast_steps)



# Evaluate the model
mse = mean_squared_error(test_data, forecast_output)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data')
plt.plot(test_data.index, forecast_output, label='Forecast', color='red')
plt.legend()
plt.show()

# Export forecast to CSV (if needed)

# Align forecast output with the corresponding dates
forecast_dates = pd.date_range(start=train_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]
forecast_df = pd.DataFrame(forecast_output, index=forecast_dates, columns=['Forecasted_Value'])

forecast_df = pd.DataFrame(forecast_output)
forecast_df.columns = ['Forecasted_Value']

# Check the DataFrame before saving
print("Forecast DataFrame:\n", forecast_df)
forecast_df.to_csv('data/arima_monthly_forecasted_values.csv')
