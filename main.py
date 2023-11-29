import numpy as np
import pandas as pd
import statsmodels.api as sm

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


# Specify the parameters for the ARIMA model
p = 1  # Order of the AR part
q = 2  # Order of the MA part
d = 0  # Differencing order (change it if your data is differenced)

# Fit the ARIMA model
model = sm.tsa.ARIMA(data['Receipt_Count'], order=(p, d, q))
results = model.fit()

# Forecast future values
future_steps = 12
forecast_output = results.forecast(steps=future_steps)

print("Forecast Output:")
print(forecast_output)

# Convert the forecast output into a DataFrame
forecast_df = pd.DataFrame(forecast_output)
# Print the DataFrame to ensure it's correctly formed
print("\nForecast DataFrame:")
print(forecast_df)
# Optionally, if you want to rename the column
forecast_df.columns = ['Forecasted_Value']

# Export to CSV
forecast_df.to_csv('forecasted_values.csv')



