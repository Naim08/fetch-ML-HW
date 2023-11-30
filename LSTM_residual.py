

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

validation_predictions = []
current_input = scaled_train_data[-n_input:]
for i in range(len(validation_data)):
    # Reshape current input for prediction
    current_input_reshaped = current_input.reshape((1, n_input, n_features))
    forecast = model.predict(current_input_reshaped, verbose=0)
    validation_predictions.append(forecast[0][0])

    # Update current_input to include the forecast and drop the oldest data point
    current_input = np.append(current_input[1:], forecast)

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

plt.title('Monthly Receipts: Actual vs Predicted W/ Residuals')
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
january_2022_date = pd.date_range(start='2022-01-01', periods=1, freq='M')
validation_df = validation_df._append({'Date': january_2022_date[0], 'Predicted Receipts': january_2022_forecast}, ignore_index=True)


plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(train_data.index, train_data['Receipt_Count'], label='Training Data', marker='o')

# Plot validation data
plt.plot(validation_data.index, validation_data['Receipt_Count'], label='Actual Validation Data', marker='o')

# Plot predicted validation data + January 2022
plt.plot(validation_df['Date'], validation_df['Predicted Receipts'], label='Predicted Data (Including Jan 2022)', linestyle='dashed', marker='x')

plt.title('Monthly Receipts: Actual vs Predicted (Including Jan 2022 W/ Residuals)')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.xticks(pd.date_range(start=train_data.index.min(), end='2022-01-31', freq='M'), rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.legend()
plt.tight_layout()
plt.show()



# Step 1: Calculate the Residuals
residuals = validation_data['Receipt_Count'] - validation_predictions.flatten()

# Step 2: Scale the Residuals
# You can use the same scaler or create a new one. Here, we'll use the same scaler.
scaled_residuals = scaler.fit_transform(residuals.values.reshape(-1, 1))

# Adjust n_input based on the size of your residuals data
n_input_adjusted = min(n_input, len(scaled_residuals) - 1)  # Ensure it's less than the length of scaled_residuals

# Create the TimeseriesGenerator with adjusted n_input
residuals_generator = TimeseriesGenerator(scaled_residuals, scaled_residuals, length=n_input_adjusted, batch_size=1)

# Step 4: Build and Train a New LSTM Model on Residuals
residual_model = Sequential()
residual_model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
residual_model.add(Dense(1))
residual_model.compile(optimizer='adam', loss='mse')

# Fit the model with the residuals data
residual_model.fit(residuals_generator, epochs=50, verbose=1)

# Step 5: Predict Residuals for the Validation Period
residuals_input = scaled_residuals[-n_input:]
residuals_input = residuals_input.reshape((1, n_input, n_features))

residuals_predictions = []
for i in range(len(validation_data)):
    forecast = residual_model.predict(residuals_input, verbose=0)
    residuals_predictions.append(forecast[0][0])
    residuals_input = np.append(residuals_input[:, 1:, :], forecast.reshape(1, 1, n_features), axis=1)

# Inverse transform the predictions to original scale
residuals_predictions = scaler.inverse_transform(np.array(residuals_predictions).reshape(-1, 1))

# Step 6: Add the Residuals Predictions to the LSTM Predictions
final_predictions = validation_predictions + residuals_predictions


if len(final_predictions) > len(validation_data):
    final_predictions = final_predictions[:len(validation_data)]
# Step 7: Evaluate the Model
mse = mean_squared_error(validation_data['Receipt_Count'], final_predictions)
mae = mean_absolute_error(validation_data['Receipt_Count'], final_predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Adjust the length of the Date array to match the length of final_predictions

adjusted_dates = validation_df['Date'][:len(final_predictions)]

# Now plot
plt.figure(figsize=(12, 6))
plt.plot(validation_data.index, validation_data['Receipt_Count'], label='Actual Validation Data', marker='o')
plt.plot(validation_df['Date'], validation_df['Predicted Receipts'], label='Predicted Validation Data (Including Jan 2022)', linestyle='dashed', marker='x')

# Ensure final_predictions and adjusted_dates are of the same length
plt.plot(adjusted_dates, final_predictions, label='Final Predictions', linestyle='dashed', marker='x')

plt.title('Monthly Receipts: Actual vs Predicted (Including Jan 2022 W/ Residuals)')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.xticks(pd.date_range(start=train_data.index.min(), end='2022-01-31', freq='M'), rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.legend()
plt.tight_layout()
plt.show()

# Save the model
residual_model.save('model/lstm_model.h5')


final_predictions = pd.DataFrame({
    'Date': validation_data.index,
    'Predicted Receipts': final_predictions.flatten()
})

final_predictions.to_csv('data/final_lstm_predictions.csv')



