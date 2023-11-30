import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

# Load and preprocess the data
data = pd.read_csv('data_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
monthly_data = data.resample('M').sum()

# Feature Engineering
monthly_data['Month'] = monthly_data.index.month

# Splitting the data
X = monthly_data[['Month']]  # Features
y = monthly_data['Receipt_Count']  # Target

# Using first 9 months for training, last 3 months for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3, shuffle=False)

# Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict for the test set
y_pred = lin_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, y, label='Actual Monthly Receipts', marker='o')
plt.plot(X_test.index, y_pred, label='Predicted Receipts (Test Data)', marker='x', linestyle='dashed')

plt.title('Linear Regression: Actual vs Predicted Receipts')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.legend()
plt.tight_layout()
plt.show()

# January corresponds to the month number 1
january_2022 = pd.DataFrame({'Month': [1]})
january_2022_forecast = lin_reg.predict(january_2022)[0]
print(f"January 2022 forecast: {january_2022_forecast}")

plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, y, label='Actual Monthly Receipts', marker='o')
plt.plot(X_test.index, y_pred, label='Predicted Receipts (Test Data)', marker='x', linestyle='dashed')

plt.title('Linear Regression: Actual vs Predicted Receipts')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.legend()
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(lin_reg, 'model/linear_regression_model.pkl')
