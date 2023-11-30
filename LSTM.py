# import numpy as np


# class MinMaxScaler:
#     """
#     Min-max scaling of the receipt counts.

#     Attributes:
#         min (float): The minimum value of the data.
#         max (float): The maximum value of the data.
#     """

#     def __init__(self):
#         self.min = None
#         self.max = None

#     def fit(self, X: np.ndarray) -> None:
#         """
#         Fit the scaler to the data.

#         Args:
#             X (np.ndarray): The input data.
#         """
#         self.min = X.min()
#         self.max = X.max()

#     def transform(self, X: np.ndarray) -> np.ndarray:
#         """
#         Transform the data using min-max scaling.

#         Args:
#             X (np.ndarray): The input data.

#         Returns:
#             np.ndarray: The scaled data.
#         """
#         return (X - self.min) / (self.max - self.min)

#     def inverse_transform(self, X: np.ndarray) -> np.ndarray:
#         """
#         Inverse transform the scaled data to the original scale.

#         Args:
#             X (np.ndarray): The scaled data.

#         Returns:
#             np.ndarray: The data in the original scale.
#         """
#         # Check if the data is a numpy array
#         if not isinstance(X, np.ndarray):
#             X = np.array(X)

#         return X * (self.max - self.min) + self.min

#     def fit_transform(self, X: np.ndarray) -> np.ndarray:
#         """
#         Fit the scaler to the data and transform it.

#         Args:
#             X (np.ndarray): The input data.

#         Returns:
#             np.ndarray: The scaled data.
#         """
#         self.fit(X)
#         return self.transform(X)



import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load and preprocess the data
data = pd.read_csv('data_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
monthly_data = data.resample('M').sum()

# Example split: use first 9 months for training and last 3 months for validation
train_data = monthly_data.iloc[:-3]
validation_data = monthly_data.iloc[-3:]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data.values)
scaled_validation_data = scaler.transform(validation_data.values)


 # Prepare the data for LSTM
n_input = 3  # Number of months to use as input
n_features = 1  # We are only using one feature

generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model with the training data
model.fit(generator, epochs=50, verbose=1)

# Predict for the validation period
validation_input = scaled_train_data[-n_input:]
validation_input = validation_input.reshape((1, n_input, n_features))

validation_predictions = []
for i in range(len(validation_data)):
    forecast = model.predict(validation_input, verbose=0)
    validation_predictions.append(forecast[0][0])
    validation_input = np.append(validation_input[:, 1:, :], forecast.reshape(1, 1, n_features), axis=1)

# Inverse transform the predictions to original scale
validation_predictions = scaler.inverse_transform(np.array(validation_predictions).reshape(-1, 1))

# Create a DataFrame for the validation predictions
validation_df = pd.DataFrame({
    'Date': validation_data.index,
    'Predicted Receipts': validation_predictions.flatten()
})



mse = mean_squared_error(validation_data['Receipt_Count'], validation_predictions)
mae = mean_absolute_error(validation_data['Receipt_Count'], validation_predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")


# Create a date range for 2022 predictions for plotting
predicted_dates = pd.date_range(start="2022-01-01", periods=12, freq='M')

plt.figure(figsize=(12, 6))

# Plot training data
plt.plot(train_data.index, train_data['Receipt_Count'], label='Training Data', marker='o')

# Plot validation data
plt.plot(validation_data.index, validation_data['Receipt_Count'], label='Actual Validation Data', marker='o')

# Plot predicted validation data
plt.plot(validation_df['Date'], validation_df['Predicted Receipts'], label='Predicted Validation Data', linestyle='dashed', marker='x')

plt.title('Monthly Receipts: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.xticks(pd.date_range(start=train_data.index.min(), end=validation_data.index.max(), freq='M'), rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.legend()
plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
plt.show()
# Predict for January 2022
january_2022_input = scaled_validation_data[-n_input:]
january_2022_input = january_2022_input.reshape((1, n_input, n_features))

january_2022_forecast = model.predict(january_2022_input, verbose=0)
january_2022_forecast = scaler.inverse_transform(january_2022_forecast)[0][0]

# Append January 2022 forecast to the validation dataframe
january_2022_date = pd.date_range(start='2022-01-01', periods=1, freq='M')
validation_df = validation_df._append({'Date': january_2022_date[0], 'Predicted Receipts': january_2022_forecast}, ignore_index=True)


plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(train_data.index, train_data['Receipt_Count'], label='Training Data', marker='o')

# Plot validation data
plt.plot(validation_data.index, validation_data['Receipt_Count'], label='Actual Validation Data', marker='o')

# Plot predicted validation data + January 2022
plt.plot(validation_df['Date'], validation_df['Predicted Receipts'], label='Predicted Data (Including Jan 2022)', linestyle='dashed', marker='x')

plt.title('Monthly Receipts: Actual vs Predicted (Including Jan 2022)')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.xticks(pd.date_range(start=train_data.index.min(), end='2022-01-31', freq='M'), rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.legend()
plt.tight_layout()
plt.show()
