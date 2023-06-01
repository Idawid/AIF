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
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


# Search query
primary_search_term = 'SPY'  # Use X as the primary search term
time_range = 'year'  # Filter posts and comments by a specific time range (in this case, one year)
sort_order = 'relevance'  # Sort the search results by relevance
query_limit = 1000  # Limit the search query to X hits
comment_depth_limit = 2  # Limit the navigation of comment tree depth to X
comments_limit_per_post = 20  # Limit the comments under post to X

# Cache unique to the search query
cache_path = 'cache'
cache_directory = f"{primary_search_term}_{time_range}_{sort_order}_limit{query_limit}_depth{comment_depth_limit}"
cache_directory = os.path.join(cache_path, cache_directory)
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)

# Cache files related to the search query
reddit_data_file = os.path.join(cache_directory, 'reddit_data.csv')
nlp_output_file = os.path.join(cache_directory, 'nlp_processed_data.csv')
reddit_sentiment_data_file = os.path.join(cache_directory, 'sentiment_analysis.csv')
date_aggregated_data_file = os.path.join(cache_directory, 'sentiments.csv')
dataset_results = os.path.join(cache_directory, 'dataset.csv')

hyperparam_results = os.path.join(cache_directory, 'hyperparams.csv')


if __name__ == '__main__':
    documents = []
    print("Retrieving query data ...")
    if os.path.isfile(reddit_data_file):
        # Retrieve from the file
        with open(reddit_data_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row

            for row in reader:
                document = tuple(row)  # Convert the row to a tuple
                documents.append(document)
        print("Done")
    else:
        print('\033[93m' + 'Query data not found!' + '\033[0m')
        print("Querying the reddit ...")
        reddit = praw.Reddit(client_id='CvDuZ97x1r9f0BnG-1_MRg', client_secret='E-y0IxK4zkRHE6ltzt-jjKcUAZi_QQ', user_agent='WSB scrapper (by /u/VultureGamer')
        subreddit = reddit.subreddit('wallstreetbets')

        for post in subreddit.search(primary_search_term, time_filter=time_range, sort=sort_order, limit=query_limit):
            # Add posts and posts' body
            post_title = post.title
            post_body = post.selftext
            post_date = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d')
            documents.append((post_title, post_date))
            documents.append((post_body, post_date))

            # Add comments
            comment_count = 0
            post.comments.replace_more(limit=comment_depth_limit)
            for comment in post.comments.list():
                if comment_count >= comments_limit_per_post:
                    break
                comment_body = comment.body
                comment_date = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d')
                documents.append((comment_body, comment_date))
                comment_count += 1
        print("Done")
        # Save scraped data to a file
        with open(reddit_data_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['title', 'date'])  # Write the header row

            for document in documents:
                writer.writerow(document)

    texts = []
    print("Retrieving normalized query data ...")
    if os.path.exists(nlp_output_file):
        # Retrieve from the file
        with open(nlp_output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                text = tuple(row)  # Convert the row to a tuple
                texts.append(text)
        print("Done")

    if not texts:
        # Process and save normalized texts to file
        print('\033[93m' + 'Normalized query data not found!' + '\033[0m')
        print("Proceeding with normalization ...")
        nlp = spacy.load('en_core_web_sm')
        for document, data in documents:
            doc = nlp(document)
            text = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            texts.append((text, data))
        print("Done")

        with open(nlp_output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['normalized_text', 'date'])  # Write the header row

            for text in texts:
                writer.writerow(text)

    sentiments = []
    print("Retrieving sentiment analysis data ...")
    if os.path.exists(reddit_sentiment_data_file):
        # Retrieve from the file
        with open(reddit_sentiment_data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                sentiment = tuple(row)  # Convert the row to a tuple
                sentiments.append(sentiment)
        print("Done")

    if not sentiments:
        for normalized_text, data in texts:
            # Calculate sentiment on normalized text
            clean_string = " ".join(eval(str(normalized_text)))
            sentiment = TextBlob(str(clean_string)).sentiment.polarity
            sentiments.append((normalized_text, float(sentiment), data))

        with open(reddit_sentiment_data_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['normalized_text', 'sentiment', 'date'])  # Write the header row

            for text in sentiments:
                writer.writerow(text)

    sentiment_data = defaultdict(list)
    combined_sentiments = []

    print("Retrieving combined sentiment data ...")
    if os.path.exists(date_aggregated_data_file):
        # Retrieve from the file
        with open(date_aggregated_data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                sentiment_data = tuple(row)  # Convert the row to a tuple
                combined_sentiments.append(sentiment_data)
        print("Done")
    else:
        print('\033[93m' + 'Combined sentiment data not found!' + '\033[0m')
        print("Combining sentiments ...")

        # Read the CSV file and collect sentiments for each date
        # Assume the previous part has completed successfully
        with open(reddit_sentiment_data_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if float(row[1]) != float(0):
                    normalized_text = row[0]
                    sentiment = float(row[1])
                    date = row[2]

                    sentiment_data[date].append(sentiment)

        for date, sentiments in sentiment_data.items():
            sentiment_average = sum(sentiments) / len(sentiments)
            sentiment_count = len(sentiments)
            combined_sentiments.append([sentiment_average, sentiment_count, date])

        # Save combined sentiments to a new CSV file
        with open(date_aggregated_data_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sentiment_average', 'sentiment_count', 'date'])  # Write the header row
            writer.writerows(combined_sentiments)

    # Historical prices + NLP dataset
    combined_data = []

    print("Retrieving dataset ...")
    if os.path.exists(dataset_results):
        # Retrieve from the file
        with open(dataset_results, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                tupled_row = tuple(row)  # Convert the row to a tuple
                combined_data.append((float(tupled_row[0]), float(tupled_row[1]), float(tupled_row[2]), tupled_row[3]))
        print("Done")
    else:
        print('\033[93m' + 'Dataset not found!' + '\033[0m')
        print("Searching for stock prices and combining with NLP ...")

        # Historical price results
        end_date = datetime.today().strftime('%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date_obj = end_date_obj - timedelta(days=365)
        start_date = start_date_obj.strftime('%Y-%m-%d')

        stock_data = yf.download(primary_search_term, start=start_date, end=end_date)
        stock_data_filled = stock_data.resample('D').interpolate()

        for date, row in stock_data_filled.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            sentiment_average = 0.0
            sentiment_count = 0
            for sentiment_avg, sentiment_cnt, sentiment_date in combined_sentiments:
                if date_str == sentiment_date:
                    sentiment_average = sentiment_avg
                    sentiment_count = sentiment_cnt
                    break
            close_price = row['Close']
            combined_data.append([float(close_price), float(sentiment_average), float(sentiment_count), date_str])

        # Save combined sentiments to a new CSV file
        with open(dataset_results, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Close Price', 'Sentiment Average', 'Sentiment Count', 'Date'])  # Write the header row
            writer.writerows(combined_data)

    # Creating a dataframe from dataset
    df_combined = pd.DataFrame(combined_data, columns=['Close Price', 'Sentiment Average', 'Sentiment Count', 'Date'])
    df_combined = df_combined.set_index('Date')



    # LSTM part
    # df_combined is DataFrame indexed by date
    df = df_combined[['Close Price', 'Sentiment Average', 'Sentiment Count']]

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
    # model = Sequential()
    # model.add(LSTM(units=40, return_sequences=True, input_shape=(x_train.shape[1], 3)))
    # model.add(LSTM(units=40, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50, return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=20))
    # model.add(Dense(units=1))   # output layer
    #
    # # Adjust the training process if needed
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=1, verbose=2)

    # model gosling ryan gosling
    # model.save('gosling.h5')

    model = load_model('gosling.h5')

    # Make the prediction
    predicted_price = model.predict(x_test)
    predicted_price = np.concatenate((predicted_price, np.zeros((len(predicted_price), 2))), axis=1)
    predicted_price = scaler.inverse_transform(predicted_price)[:, 0]

    # Visualize the prediction
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(df['Close Price'], label='Actual Close Price')
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
