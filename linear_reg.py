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

# Load and preprocess the data
data = pd.read_csv('data_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
monthly_data = data.resample('M').sum()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data.values)

 # Prepare the data for LSTM
n_input = 3  # Number of months to use as input
n_features = 1  # We are only using one feature

generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(generator, epochs=50, verbose=1)


# Predict for the next 12 months
forecast_input = scaled_data[-n_input:]
forecast_input = forecast_input.reshape((1, n_input, n_features))

predictions = []
for i in range(12):
    forecast = model.predict(forecast_input, verbose=0)
    predictions.append(forecast[0][0])
    forecast_input = np.append(forecast_input[:, 1:, :], [[forecast]], axis=1)

# Inverse transform the predictions to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'Month': range(1, 13),
    'Predicted Receipts': predictions.flatten()
})


# Display the predictions DataFrame
plt.figure(figsize=(12, 6))
plt.plot(monthly_data.index, monthly_data['Receipt_Count'], label='Actual Receipts')
plt.plot(predictions_df['Month'], predictions_df['Predicted Receipts'], label='Forecasted Receipts', color='red')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.title('Monthly Receipts Forecast for 2022')
plt.legend()
plt.show()
