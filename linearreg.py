import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# Load the data
data = pd.read_csv('data_daily.csv', parse_dates=['# Date'], index_col='# Date')
data.index.name = 'Date'
# Assuming 'data' has a date column and a target variable, e.g., 'receipt_count'
data['Date'] = np.arange(len(data))  # Create a time index

# Prepare features (X) and target (y)
X = data[['Date']]
y = data['Receipt_Count']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

num_future_days = 30

# Forecast future values
future_time_index = np.arange(len(data), len(data) + num_future_days)
future_predictions = model.predict(future_time_index.reshape(-1, 1))

# future_predictions contains the forecasted values for 'num_future_days' ahead

# Optional: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['Receipt_Count'], label='Original Data')

plt.plot(future_time_index, future_predictions, label='Forecasted Values')
plt.legend()
plt.show()
