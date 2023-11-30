import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.api as sm

def forecast_sarima(data, ar_coefficients, ma_coefficients, seasonal_ar_coefficients, seasonal_ma_coefficients, p, q, P, Q, S, steps):
    """
    Forecast future values using a SARIMA model.
    """
    forecast = []
    data_extended = list(data.values.flatten())  # Flatten the data to a list for easier manipulation
    errors = [0] * max(q, Q * S)  # Initialize errors for the MA component

    for t in range(steps):
        # AR Component
        lag_values = data_extended[-p:]
        forecast_value = ar_coefficients[0] + np.sum(ar_coefficients[1:] * lag_values)

        # Seasonal AR Component
        if P > 0:
            seasonal_lag_values = [data_extended[-i] for i in range(S, P * S + 1, S)]
            forecast_value += np.sum(seasonal_ar_coefficients * seasonal_lag_values)

        # MA Component
        forecast_value += np.sum(ma_coefficients * errors[-q:])

        # Seasonal MA Component
        if Q > 0:
            seasonal_errors = errors[-Q * S:]
            forecast_value += np.sum(seasonal_ma_coefficients * seasonal_errors)

        # Update errors
        actual_value = data.iloc[t].values[0] if t < len(data) else 0  # Use iloc for accessing data by index
        error = actual_value - forecast_value
        errors.append(error)

        forecast.append(forecast_value)
        data_extended.append(forecast_value)

    return forecast


data = pd.read_csv('data_daily.csv', parse_dates=['Date'], index_col='Date')

train_data = data[data.index < '2021-10-01']
test_data = data[data.index >= '2021-10-01']


# Fit the SARIMA model
p, d, q = 1, 1, 1  # Non-seasonal parameters
P, D, Q, S = 1, 1, 1, 12 # Seasonal parameters (example: yearly seasonality with S=12)



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


# Modify this part to include seasonal parameters
# For simplicity, let's assume some dummy coefficients for seasonal parts
seasonal_ar_coefficients = np.array([0.5]) # Placeholder values
seasonal_ma_coefficients = np.array([0.5]) # Placeholder values

forecast_steps = len(test_data)
forecast_output = forecast_sarima(train_data,
                                  results.arparams,
                                  results.maparams,
                                  seasonal_ar_coefficients,
                                  seasonal_ma_coefficients,
                                  p,
                                  len(results.maparams),
                                  P,
                                  Q,
                                  S,
                                  forecast_steps)

# Evaluate the model
mse = mean_squared_error(test_data, forecast_output)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Plotting the results

# Export forecast to CSV (if needed)
forecast_dates = pd.date_range(start=train_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]
forecast_df = pd.DataFrame(forecast_output, index=forecast_dates, columns=['Forecasted_Value'])
forecast_df.to_csv('data/sarima_forecast_output.csv')
