# main.py

from ensemble import (load_and_preprocess_data, build_lstm_model, train_model,
                      predict_with_model, calculate_residuals, plot_results,
                      save_predictions, evaluate_model)

# Load and preprocess the data
train_data, validation_data = load_and_preprocess_data('data_daily.csv')

# Build and train the initial LSTM model
initial_model = build_lstm_model(n_input=3, n_features=1)
initial_model, initial_scaler = train_model(initial_model, train_data, n_input=3, n_features=1)

# Make initial predictions
initial_predictions = predict_with_model(initial_model, initial_scaler, validation_data, n_input=3, n_features=1)

# Calculate residuals
residuals = calculate_residuals(validation_data['Receipt_Count'], initial_predictions)

# Build and train the LSTM model for residuals
residual_model = build_lstm_model(n_input=3, n_features=1)
residual_model, residual_scaler = train_model(residual_model, pd.DataFrame(residuals), n_input=3, n_features=1)

# Predict residuals
residual_predictions = predict_with_model(residual_model, residual_scaler, pd.DataFrame(residuals), n_input=3, n_features=1)

# Add the residuals predictions to the initial predictions
final_predictions = initial_predictions + residual_predictions

# Evaluate the model
evaluate_model(validation_data['Receipt_Count'], final_predictions)

# Plot the results
plot_results(validation_data, final_predictions, 'Monthly Receipts: Actual vs Predicted (Including Residuals)')

# Save the final
save_predictions(validation_data, final_predictions, 'data/final_lstm_predictions.csv')
