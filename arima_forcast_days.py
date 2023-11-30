import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import itertools

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
data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col='# Date')

# Assuming data is already preprocessed (e.g., differenced if needed)
# p = 2  # Order of the AR part
# q = 2  # Order of the MA part
# ar_coefficients = fit_ar_model(data['Receipt_Count'], p)
# ma_coefficients = np.random.rand(q)  # Placeholder for MA coefficients

# # Forecast future values
# future_steps = 12
# forecasted_values = forecast_arima(data['Receipt_Count'], ar_coefficients, ma_coefficients, p, q, future_steps)

# # Convert the forecasted values to a DataFrame
# forecast_df = pd.DataFrame(forecasted_values, columns=['Forecasted_Value'])


# forecast_df.index = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='D')[1:]
# forecast_df.to_csv('forecasted_values.csv')

# Optional: Generate ACF and PACF plots if you want to visually inspect
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data['Receipt_Count'], ax=plt.gca(), lags=40)
plt.subplot(122)
plot_pacf(data['Receipt_Count'], ax=plt.gca(), lags=40)
#plt.show()

# Split the data into training and test sets
train_data = data[data.index < '2021-10-01']
test_data = data[data.index >= '2021-10-01']

# Find the best 'q' value based on AIC (using training data)
best_aic = np.inf
best_q = 0
p = 2 # Assuming you've determined p is 1
d = 0 # Differencing order

for q in range(10):  # Testing q values from 0 to 4
    try:
        model = sm.tsa.ARIMA(train_data['Receipt_Count'], order=(p, d, q))
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_q = q
    except:  # Handle cases where the model fails to converge
        continue

print(f"Best q value: {best_q} with AIC: {best_aic}")

# Fit the ARIMA model with the best 'q' value found
model = sm.tsa.ARIMA(train_data['Receipt_Count'], order=(p, d, q))
results = model.fit()

# # Forecast future values
forecast_steps = len(test_data)
forecast_output = results.forecast(steps=forecast_steps)

# Convert the forecast output into a DataFrame
forecast_df = pd.DataFrame(forecast_output)
forecast_df.columns = ['Forecasted_Value']
# Evaluate the model
mse = mean_squared_error(test_data['Receipt_Count'], forecast_output)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

forecast_df.to_csv('data/forecasted_values_days.csv')

future_steps = 120
forecast_output = results.forecast(steps=future_steps)

# Convert the forecast output into a DataFrame
forecast_df = pd.DataFrame(forecast_output)
forecast_df.columns = ['Forecasted_Value']

# Export to CSV
forecast_df.to_csv('data/forecasted_values_days_2022.csv')


plt.figure(figsize=(12, 6))
plt.plot(train_data['Receipt_Count'], label='Training Data')
plt.plot(test_data['Receipt_Count'], label='Test Data')
plt.plot(forecast_output, label='Forecast', color='red')
plt.legend()
plt.show()
