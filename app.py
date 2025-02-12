import warnings
warnings.filterwarnings('ignore', category=Warning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
from sentiment import get_news_articles, analyze_sentiment_of_articles, generate_signal
plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Create a custom LSTM layer that ignores the time_major parameter
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove time_major if present
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# Register the custom layer
with tf.keras.utils.custom_object_scope({'LSTM': CustomLSTM}):
    model = load_model('stock_dl_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Create static directory if it doesn't exist
            os.makedirs('static', exist_ok=True)
            
            stock = request.form.get('stock')
            if not stock:
                stock = 'POWERGRID.NS'  # Default stock if none is entered
            
            # Define the start and end dates for stock data
            start = dt.datetime(2000, 1, 1)
            end = dt.datetime(2024, 10, 1)
            
            # Download stock data
            df = yf.download(stock, start=start, end=end)
            
            # Descriptive Data
            data_desc = df.describe()
            
            # Exponential Moving Averages
            ema20 = df.Close.ewm(span=20, adjust=False).mean()
            ema50 = df.Close.ewm(span=50, adjust=False).mean()
            ema100 = df.Close.ewm(span=100, adjust=False).mean()
            ema200 = df.Close.ewm(span=200, adjust=False).mean()
            
            # Data splitting
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
            
            # Scaling data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)
            
            # Prepare data for prediction
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)
            
            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])
            x_test, y_test = np.array(x_test), np.array(y_test)

            # Make predictions
            y_predicted = model.predict(x_test)
            
            # Inverse scaling for predictions
            scaler = scaler.scale_
            scale_factor = 1 / scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor
            
            # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
            plt.clf()  # Clear any existing plots
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df.Close, 'y', label='Closing Price')
            ax1.plot(ema20, 'g', label='EMA 20')
            ax1.plot(ema50, 'r', label='EMA 50')
            ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")
            ax1.legend()
            ema_chart_path = os.path.join("static", "ema_20_50.png")
            plt.savefig(ema_chart_path)
            plt.close(fig1)
            
            # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
            plt.clf()  # Clear any existing plots
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df.Close, 'y', label='Closing Price')
            ax2.plot(ema100, 'g', label='EMA 100')
            ax2.plot(ema200, 'r', label='EMA 200')
            ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Price")
            ax2.legend()
            ema_chart_path_100_200 = os.path.join("static", "ema_100_200.png")
            plt.savefig(ema_chart_path_100_200)
            plt.close(fig2)
            
            # Plot 3: Prediction vs Original Trend
            plt.clf()  # Clear any existing plots
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
            ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
            ax3.set_title("Prediction vs Original Trend")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Price")
            ax3.legend()
            prediction_chart_path = os.path.join("static", "stock_prediction.png")
            plt.savefig(prediction_chart_path)
            plt.close(fig3)
            
            # Clean up
            plt.close('all')
            
            # Save dataset as CSV
            csv_file_path = os.path.join("static", f"{stock}_dataset.csv")
            df.to_csv(csv_file_path)
            
            # Add sentiment analysis
            api_key = '84304d30650d4d959fee116667241dda'
            articles = get_news_articles(stock, api_key)
            sentiment_scores = analyze_sentiment_of_articles(articles)
            _, avg_sentiment = generate_signal(sentiment_scores)  # Using _ to ignore the signal
            
            return render_template('index.html',
                                plot_path_ema_20_50=ema_chart_path,
                                plot_path_ema_100_200=ema_chart_path_100_200,
                                plot_path_prediction=prediction_chart_path,
                                data_desc=data_desc.to_html(classes='table table-bordered'),
                                dataset_link=csv_file_path,
                                sentiment_score=f"{avg_sentiment:.2f}",
                                stock_symbol=stock)
                                
        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
