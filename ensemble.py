import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and preprocess the data (make sure this matches your original preprocessing)
data = pd.read_csv('data_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
monthly_data = data.resample('M').sum()
train_data = monthly_data.iloc[:-3]  # Assuming first 9 months for training
validation_data = monthly_data.iloc[-3:]  # Assuming last 3 months for validation

# Load the models
lstm_model = load_model('model/lstm_model.h5')
lin_model = joblib.load('model/linear_regression_model.pkl')

# Scale the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_data.values)
scaled_validation_data = scaler.transform(validation_data.values)

# Prepare validation data for LSTM prediction
n_input = 3
n_features = 1
validation_input = scaled_data[-(n_input + len(validation_data)):-len(validation_data)]
validation_input = validation_input.reshape((1, validation_input.shape[0], n_features))

# Generate predictions from LSTM
lstm_predictions = lstm_model.predict(validation_input).flatten()
lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()

# Prepare validation data for Linear Regression prediction
#X_test = validation_data.index.month.reshape(-1, 1)
X_test = validation_data.index.month.values.reshape(-1, 1)
# Generate predictions from Linear Regression
lin_predictions = lin_model.predict(X_test)

# Combine the predictions
combined_predictions = (lstm_predictions + lin_predictions) / 2

# Evaluate the combined model
combined_mse = mean_squared_error(validation_data['Receipt_Count'], combined_predictions)
combined_mae = mean_absolute_error(validation_data['Receipt_Count'], combined_predictions)
print(f"Combined Mean Squared Error: {combined_mse}")
print(f"Combined Mean Absolute Error: {combined_mae}")

print(f"LSTM Predictions: {lstm_predictions}")
print(f"Linear Regression Predictions: {lin_predictions}")
# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Receipt_Count'], label='Training Data')
plt.plot(validation_data.index, validation_data['Receipt_Count'], label='Actual Receipts', marker='o')
plt.plot(validation_data.index, combined_predictions, label='Combined Predictions', marker='x', linestyle='dashed')
plt.plot(validation_data.index, lin_predictions, label='LR Predictions', marker='x', linestyle='dashed')
plt.title('Combined Model Predictions vs Actual Receipts')
plt.xlabel('Date')
plt.ylabel('Receipts')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.legend()
plt.tight_layout()
plt.show()

combined_predictions = pd.DataFrame({
    'Date': validation_data.index,
    'Predicted Receipts': combined_predictions
})

combined_predictions.to_csv('data/combined_predictions.csv')
