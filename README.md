📈 Stock Price Prediction using LSTM and Linear Regression

This project predicts the future prices of Cochin Shipyard Limited (COCHINSHIP.NS) stock using historical market data from Yahoo Finance.
It combines Linear Regression and LSTM (Long Short-Term Memory) deep learning models to analyze trends and forecast prices.
The project also includes a Streamlit web app for interactive visualization.

🚀 Features

✅ Yahoo Finance Integration – Automatically fetches live stock data.

📊 7-Day Moving Average – Adds trend-based feature for better model understanding.

⚙️ Linear Regression Model – Provides a baseline predictive analysis.

🧠 LSTM Neural Network – Captures sequential dependencies in time-series data.

📉 Model Evaluation – Calculates RMSE (Root Mean Squared Error) for accuracy.

🌐 Streamlit Web Interface – Interactive visualization of predictions vs. actual prices.

🧾 Project Workflow
1. Data Collection

Data is downloaded using the yfinance library for the ticker symbol COCHINSHIP.NS (Cochin Shipyard Ltd).

data = yf.download("COCHINSHIP.NS", start="2022-01-01", end="2024-10-01")

2. Data Preprocessing

Missing values are forward-filled.

A 7-day moving average (MA7) is computed.

Data is scaled between 0–1 using MinMaxScaler.

3. Modeling

Linear Regression: Used as a baseline model to correlate MA7 with Close prices.

LSTM Model: Built with TensorFlow/Keras using a 2-layer LSTM architecture to capture time-series trends.

4. Evaluation

Model performance is measured using Root Mean Squared Error (RMSE).

5. Visualization

Plots are generated using Matplotlib.

Interactive charts and metrics are displayed in the Streamlit app.

🧩 Technologies Used
Category	Tools/Libraries
Data Source	Yahoo Finance
 via yfinance
Data Handling	pandas, numpy
Machine Learning	scikit-learn
Deep Learning	TensorFlow, Keras
Visualization	matplotlib, streamlit
Metrics	RMSE (Root Mean Squared Error)
🧠 LSTM Model Architecture
Input → LSTM(60, return_sequences=True) → LSTM(60) → Dense(1)


Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

Epochs: 20

Batch Size: 32

📊 Results

The model predicts the trend of stock prices reasonably well.

RMSE is printed and displayed in the Streamlit interface.

Interactive line charts allow visual comparison of predicted vs. actual stock prices.

🧮 Example Output

RMSE (Root Mean Squared Error): ~x.xx (varies by run)

Chart:

Purple line → Actual Prices

Red dashed line → Predicted Prices

🖥️ Running the Streamlit App
1. Clone the Repository
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction

2. Install Dependencies
pip install -r requirements.txt

3. Run the App
streamlit run app.py

4. View in Browser

Visit http://localhost:8501/
 to explore the dashboard.

📂 Project Structure
📁 stock-price-prediction/
│
├── app.py                  # Main Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
└── data/                   # Optional folder for saving fetched data

⚡ Future Improvements

🔮 Add ARIMA or Prophet models for comparison

📆 Extend predictions to multiple future days

🧩 Integrate more technical indicators (RSI, MACD, Bollinger Bands)

📱 Deploy on Streamlit Cloud or Hugging Face Spaces

🏷️ Author

👤 Navadeeep
B.Tech CSE | Machine Learning & AI Enthusiast
