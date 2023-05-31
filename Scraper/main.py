import os
from collections import defaultdict

import praw
import spacy
import csv
from datetime import datetime, timedelta
from textblob import TextBlob
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Search query
search_query = 'NVDA'   # Use X as the primary search term
time_filter = 'year'    # Filter posts and comments by a specific time range (in this case, one year)
sort_by = 'relevance'   # Sort the search results by relevance
limit = 100             # Limit the search query to X hits
comment_tree_depth = 1  # Limit the navigation of comment tree depth to X
comments_per_post = 2   # Limit the comments under post to X

# Cache unique to the search query
cache_path = 'cache'
cache_directory = f"{search_query}_{time_filter}_{sort_by}_limit{limit}_depth{comment_tree_depth}"
cache_directory = os.path.join(cache_path, cache_directory)
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)

# Cache files related to the search query
query_results = os.path.join(cache_directory, 'raw_scraped_data.csv')
normalized_query_results = os.path.join(cache_directory, 'normalized_scraped_data.csv')
hyperparam_results = os.path.join(cache_directory, 'hyperparams.csv')
sentiment_analysis_results = os.path.join(cache_directory, 'raw_sentiment_scores.csv')
combined_sentiments_results = os.path.join(cache_directory, 'sentiments.csv')


if __name__ == '__main__':
    documents = []
    print("Retrieving query data ...")
    if os.path.isfile(query_results):
        # Retrieve from the file
        with open(query_results, 'r', encoding='utf-8') as csvfile:
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

        for post in subreddit.search(search_query, time_filter=time_filter, sort=sort_by, limit=limit):
            # Add posts and posts' body
            post_title = post.title
            post_body = post.selftext
            post_date = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d')
            documents.append((post_title, post_date))
            documents.append((post_body, post_date))

            # Add comments
            comment_count = 0
            post.comments.replace_more(limit=comment_tree_depth)
            for comment in post.comments.list():
                if comment_count >= comments_per_post:
                    break
                comment_body = comment.body
                comment_date = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d')
                documents.append((comment_body, comment_date))
                comment_count += 1
        print("Done")
        # Save scraped data to a file
        with open(query_results, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['title', 'date'])  # Write the header row

            for document in documents:
                writer.writerow(document)

    texts = []
    print("Retrieving normalized query data ...")
    if os.path.exists(normalized_query_results):
        # Retrieve from the file
        with open(normalized_query_results, 'r', encoding='utf-8') as f:
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

        with open(normalized_query_results, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['normalized_text', 'date'])  # Write the header row

            for text in texts:
                writer.writerow(text)

    sentiments = []
    print("Retrieving sentiment analysis data ...")
    if os.path.exists(sentiment_analysis_results):
        # Retrieve from the file
        with open(sentiment_analysis_results, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                sentiment = tuple(row)  # Convert the row to a tuple
                sentiments.append(sentiment)
        print("Done")

    if not sentiments:
        for normalized_text, data in texts:
            # Calculate sentiment on normalized text
            clean_string = " ".join(eval(normalized_text))
            sentiment = TextBlob(str(clean_string)).sentiment.polarity
            sentiments.append((normalized_text, float(sentiment), data))

        with open(sentiment_analysis_results, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['normalized_text', 'sentiment', 'date'])  # Write the header row

            for text in sentiments:
                writer.writerow(text)

    sentiment_data = defaultdict(list)
    combined_sentiments = []

    print("Retrieving combined sentiment data ...")
    if os.path.exists(combined_sentiments_results):
        # Retrieve from the file
        with open(combined_sentiments_results, 'r', encoding='utf-8') as f:
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
        with open(sentiment_analysis_results, 'r', encoding='utf-8') as csvfile:
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
            combined_sentiments.append([sentiment_average, sentiment_count, date])\

        # Save combined sentiments to a new CSV file
        with open(combined_sentiments_results, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sentiment_average', 'sentiment_count', 'date'])  # Write the header row
            writer.writerows(combined_sentiments)


    # Historical price results
    end_date = datetime.today().strftime('%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    start_date_obj = end_date_obj - timedelta(days=365)
    start_date = start_date_obj.strftime('%Y-%m-%d')

    stock_data = yf.download(search_query, start=start_date, end=end_date)
    stock_data_filled = stock_data.resample('D').interpolate()

    combined_data = []
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

    df_combined = pd.DataFrame(combined_data, columns=['Close Price', 'Sentiment Average', 'Sentiment Count', 'Date'])
    df_combined = df_combined.set_index('Date')


    def adf_test(series, signif=0.05):
        dftest = adfuller(series, autolag='AIC')
        adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags', '# Observations'])
        for key, value in dftest[4].items():
            adf['Critical Value (%s)' % key] = value
        print(adf)

        p = adf['p-value']
        if p <= signif:
            print(f" Series is Stationary")
        else:
            print(f" Series is Non-Stationary")


    # apply adf test on the series
    adf_test(df_combined["Close Price"])
    adf_test(df_combined["Sentiment Average"])
    adf_test(df_combined["Sentiment Count"])
