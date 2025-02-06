import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# Step 1: Download data from Yahoo Finance for the stock
data = yf.download("COCHINSHIP.NS", start="2022-01-01", end="2024-10-01")

# Step 2: Forward fill any missing values in the dataset
data = data.fillna(method='ffill')

# Step 3: Create a 7-day moving average column (MA7)
data['MA7'] = data['Close'].rolling(window=7).mean()

# Step 4: Drop rows with NaN values (caused by the rolling average)
data = data.dropna()

# Step 5: Plot the Closing Price and Moving Average
plt.figure(figsize=(14, 10))
plt.plot(data['Close'], label="Closing Price", color='purple')
plt.plot(data['MA7'], label="7-Day Moving Average", color='Red', linestyle='--')
plt.title("Stock Closing Price and 7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Apply MinMax Scaling to the 'Close' price
scaler = MinMaxScaler()
data['Scaled_Close'] = scaler.fit_transform(data[['Close']])

# Step 7: Prepare data for Linear Regression (MA7 as feature, Scaled_Close as target)
X = data[['MA7']]
y = data['Scaled_Close']

# Step 8: Fit the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Step 9: Display model coefficients
print(f"Linear Model Coefficient: {linear_model.coef_[0]}")
print(f"Linear Model Intercept: {linear_model.intercept_}")

# Step 10: Prepare data for LSTM (scaled close prices)
scaled_close = data['Scaled_Close'].values

# Step 11: Create sequences for LSTM (sequence length = 60)
sequence_length = 60  # Number of time steps to consider
X_lstm, y_lstm = [], []

for i in range(sequence_length, len(scaled_close)):
    X_lstm.append(scaled_close[i-sequence_length:i])  # Use last 'sequence_length' prices
    y_lstm.append(scaled_close[i])  # Predict the next value (next time step)

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Step 12: Split into training and testing sets (80-20 split)
split_index = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:split_index], X_lstm[split_index:]
y_train, y_test = y_lstm[:split_index], y_lstm[split_index:]

# Step 13: Reshape for LSTM input (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 14: Define and compile the LSTM model
lstm_model = Sequential([
    LSTM(60, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(60),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Step 15: Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)

# Step 16: Predict on the test set
y_pred = lstm_model.predict(X_test)

# Step 17: Rescale predictions back to the original price scale
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 18: Calculate RMSE
rmse = mean_squared_error(y_test_rescaled, y_pred_rescaled, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 19: Plot predictions vs. actual values
plt.figure(figsize=(14, 10))
plt.plot(y_test_rescaled, label="Actual Prices", color='purple')
plt.plot(y_pred_rescaled, label="Predicted Prices", color='Red', linestyle='--')
plt.title("LSTM Model: Actual vs Predicted Prices")
plt.xlabel("Time")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.show()

# Streamlit code for displaying the results
st.title("Stock Price Prediction")

# Show the actual stock price data
st.line_chart(data['Close'])

# Display RMSE as a text on Streamlit app
st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Show the predictions vs actual prices plot in the Streamlit app
st.subheader("Predictions vs Actual Prices")
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(y_test_rescaled, label="Actual Prices", color='purple')
ax.plot(y_pred_rescaled, label="Predicted Prices", color='Red', linestyle='--')
ax.set_title("LSTM Model: Actual vs Predicted Prices")
ax.set_xlabel("Time")
ax.set_ylabel("Price (INR)")
ax.legend()
ax.grid(True)

# Show the plot in the Streamlit app
st.pyplot(fig)
