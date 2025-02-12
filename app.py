import warnings
warnings.filterwarnings('ignore', category=Warning)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
from sentiment import get_news_articles, analyze_sentiment_of_articles, generate_signal

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
            end = dt.datetime.now()  # This will get data up to today
            
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
            
            # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df.Close, name='Closing Price', line=dict(color='yellow')))
            fig2.add_trace(go.Scatter(x=df.index, y=ema100, name='EMA 100', line=dict(color='green')))
            fig2.add_trace(go.Scatter(x=df.index, y=ema200, name='EMA 200', line=dict(color='red')))
            fig2.update_layout(
                title="Closing Price vs Time (100 & 200 Days EMA)",
                xaxis_title="Time",
                yaxis_title="Price",
                template='plotly_dark'
            )
            plot_path_ema_100_200 = fig2.to_html(full_html=False)

            # Plot 3: Prediction vs Original Trend
            fig3 = go.Figure()
            
            # Create date range for x-axis
            prediction_dates = df.index[-(len(y_test)):]
            
            # Create future dates for prediction
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=2, freq='B')
            
            # Get the last 100 days of data for future prediction
            last_100_days = df['Close'].tail(100).values.reshape(-1, 1)
            
            # Scale the last 100 days using the same scaling factor as before
            last_100_days_scaled = last_100_days / scale_factor
            
            # Prepare input for future prediction
            X_future = last_100_days_scaled.reshape(1, 100, 1)
            
            # Make future predictions
            future_pred = model.predict(X_future)
            
            # Scale back predictions using the same scale factor
            future_pred = future_pred * scale_factor
            
            # Get the last actual prediction to ensure continuity
            last_prediction = y_predicted[-1]
            
            # Adjust future predictions to maintain trend from last prediction
            adjustment = last_prediction - future_pred[0][0]
            future_pred = future_pred + adjustment
            
            # Combine current and future predictions
            all_dates = prediction_dates.union(future_dates)
            extended_predictions = np.append(y_predicted.flatten(), future_pred.flatten())
            
            # Plot original price (green line)
            fig3.add_trace(go.Scatter(
                x=prediction_dates,
                y=y_test,
                name='Original Price',
                line=dict(color='green', width=2)
            ))
            
            # Plot predicted price including future (red line)
            fig3.add_trace(go.Scatter(
                x=all_dates,
                y=extended_predictions,
                name='Predicted Price',
                line=dict(color='red', width=2)
            ))
            
            fig3.update_layout(
                title="Prediction vs Original Trend",
                xaxis_title="Time",
                yaxis_title="Price",
                template='plotly_dark',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            plot_path_prediction = fig3.to_html(full_html=False)

            # Save dataset as CSV
            csv_file_path = os.path.join("static", f"{stock}_dataset.csv")
            df.to_csv(csv_file_path)
            
            # Add sentiment analysis
            api_key = '84304d30650d4d959fee116667241dda'
            articles = get_news_articles(stock, api_key)
            sentiment_scores = analyze_sentiment_of_articles(articles)
            _, avg_sentiment = generate_signal(sentiment_scores)

            return render_template('index.html',
                                plot_path_ema_100_200=plot_path_ema_100_200,
                                plot_path_prediction=plot_path_prediction,
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
