import os
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from config import cache_directory


def load_dataset():
    dataset_file = os.path.join(cache_directory, 'dataset_file.csv')
    if os.path.isfile(dataset_file):
        df = pd.read_csv(dataset_file, index_col='date', usecols=['date', 'close_price', 'average_sentiment', 'sentiment_count'])
        df = df[['close_price', 'average_sentiment', 'sentiment_count']]
    else:
        print(Fore.LIGHTRED_EX + "Could not retrieve the dataset." + Style.RESET_ALL)
        quit()
    return df


def create_dataset(start_index, end_index, scaled_data):
    x_data, y_data = [], []
    for i in range(start_index, end_index):
        x_data.append(scaled_data[i - 60:i, :])     # Based on the 60 previous values
        y_data.append(scaled_data[i, 0])            # we predict the value at the current point in time
    x_data, y_data = np.array(x_data), np.array(y_data)

    # Reshape the data into 3-D array
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 3))

    return x_data, y_data


def build_model(input_shape):
    # Create and fit the LSTM network. Adjust layers if needed
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=50))
    model.add(Dense(units=1))
    return model


def train_model(model, x_train, y_train, x_val, y_val):
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=1, verbose=2)



def main():
    df = load_dataset()
    dataset = df.values

    split_train = 70 / 100  # Training: 70%
    split_val = 20 / 100  # Validation: 20%
    split_test = 10 / 100  # Testing: 10%

    len_train = int(dataset.shape[0] * split_train)
    len_valid = int(dataset.shape[0] * split_val)
    len_test = dataset.shape[0] - len_train - len_valid

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = create_dataset(60, len_train, scaled_data)
    x_val, y_val = create_dataset(len_train, len_train + len_valid, scaled_data)
    x_test, y_test = create_dataset(len_train + len_valid, len_train + len_valid + len_test, scaled_data)

    model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]))

    train_model(model, x_train, y_train, x_val, y_val)

    # model gosling ryan gosling
    model.save('gosling.h5')

    #model = load_model('gosling.h5')

    # Evaluate the model's performance after it has been trained
    predictions = model.predict(x_test)

    predictions = scaler.inverse_transform(
        np.concatenate([predictions.reshape(-1, 1), np.zeros((len(predictions), 2))], axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 2))], axis=1))[:, 0]

    mse = np.mean((predictions - y_test) ** 2)
    print('Test MSE: ', mse)

    # Visualize the prediction
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(df['close_price'], label='Actual Close Price')
    plt.plot(df.iloc[len_train + len_valid:].index, predictions, label='Predicted Close Price')
    plt.legend()
    plt.show()

    last_x_days = scaled_data[-60:]
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


if __name__ == '__main__':
    main()