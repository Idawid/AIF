import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict

import praw
import spacy
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from config import cache_directory


if __name__ == '__main__':

    dataset_file = os.path.join(cache_directory, 'dataset_file.csv')

    if os.path.isfile(dataset_file):
        df = pd.read_csv(dataset_file)
    else:
        print(Fore.LIGHTRED_EX + "Could not retrieve the dataset." + Style.RESET_ALL)
        quit()

    # LSTM part
    # df_combined is DataFrame indexed by date
    df = df.set_index('date')
    df = df[['close_price', 'average_sentiment', 'sentiment_count']]

    dataset = df.values
    split_train = 70 / 100  # Training: 70%
    split_val = 20 / 100  # Validation: 20%
    split_test = 10 / 100  # Testing: 10%

    train = dataset[:int(dataset.shape[0] * split_train)]
    valid = dataset[int(dataset.shape[0] * split_train): int(dataset.shape[0] * (split_train + split_val))]
    test = dataset[int(dataset.shape[0] * (split_train + split_val)):]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, :])
        y_train.append(scaled_data[i, 0])  # predict the 'Close Price'
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data into 3-D array
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))

    # Create the validation data set
    x_val, y_val = [], []
    for i in range(len(train), len(train) + len(valid)):
        x_val.append(scaled_data[i - 60:i, :])
        y_val.append(scaled_data[i, 0])
    x_val, y_val = np.array(x_val), np.array(y_val)

    # Reshape the data into 3-D array
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 3))

    # Create the testing data set
    x_test, y_test = [], []
    for i in range(len(train) + len(valid), len(dataset)):
        x_test.append(scaled_data[i - 60:i, :])
        y_test.append(scaled_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Reshape the data into 3-D array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))

    # Create and fit the LSTM network. Adjust layers if needed
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 3)))
    model.add(LSTM(units=200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=150))
    model.add(Dense(units=1))   # output layer

    # Adjust the training process if needed
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=1, verbose=2)

    #model gosling ryan gosling
    model.save('gosling.h5')

    #model = load_model('gosling.h5')

    # Make the prediction
    predicted_price = model.predict(x_test)
    predicted_price = np.concatenate((predicted_price, np.zeros((len(predicted_price), 2))), axis=1)
    predicted_price = scaler.inverse_transform(predicted_price)[:, 0]

    # Visualize the prediction
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(df['close_price'], label='Actual Close Price')
    plt.plot(df.iloc[len(train)+len(valid):].index, predicted_price, label='Predicted Close Price')
    plt.legend()
    plt.show()


    last_x_days = scaled_data[-30:]
    next_month_prices = []

    predict_for_x_days = 7
    for _ in range(predict_for_x_days):  # Predict the next X days
        # Reshape and expand dims to fit the model input shape
        last_x_days = np.expand_dims(last_x_days, axis=0)

        # Predict the next day price
        next_day_price = model.predict(last_x_days)

        next_day_price = np.concatenate((next_day_price, np.zeros((len(next_day_price), 2))), axis=1)
        # Append the predicted price to the end of sequence and use the last 60 days for next prediction
        last_x_days = np.concatenate((last_x_days[0][1:], next_day_price), axis=0)

        # Store the predicted price
        next_month_prices.append(scaler.inverse_transform(next_day_price)[:, 0][0])

    print(next_month_prices)

    # Plot the predicted close prices for the next month
    plt.figure(figsize=(16, 8))
    plt.title('Predicted Close Prices for Next Month')
    plt.xlabel('Day', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(next_month_prices, label='Predicted Close Price for the next month')
    plt.legend()
    plt.show()
