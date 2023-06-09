import os

import pandas as pd
import yfinance as yf
import praw
import spacy
from tqdm import tqdm
from colorama import Fore, Style
from textblob import TextBlob
from datetime import datetime, timedelta
from config import query_parameters, cache_directory, create_cache_dir


def query_reddit_data(query_params, cache_base_dir='cache'  ):
    reddit = praw.Reddit(client_id=query_params['client_id'],
                         client_secret=query_params['client_secret'],
                         user_agent=query_params['user_agent'])

    subreddit = reddit.subreddit(query_params['subreddit'])

    documents = []
    posts = list(subreddit.search(query_params['search_term'], time_filter=query_params['time_range'],
                                  sort=query_params['sort_order'], limit=query_params['post_limit']))
    for post in tqdm(posts, desc="Scraping Reddit", unit="post"):
        # Add posts and posts' body
        post_title = post.title
        post_body = post.selftext
        post_date = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d')
        documents.append((post_title, post_date))
        documents.append((post_body, post_date))

        # Add comments
        comment_count = 0
        post.comments.replace_more(limit=query_params['comment_depth'])
        for comment in post.comments.list():
            if comment_count >= query_params['comment_limit']:
                break
            comment_body = comment.body
            comment_date = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d')
            documents.append((comment_body, comment_date))
            comment_count += 1

    df = pd.DataFrame(documents, columns=['text', 'date'])
    return df


def process_text(df):
    nlp = spacy.load('en_core_web_sm')
    df['normalized_text'] = df['text'].apply(lambda doc: [token.lemma_ for token in nlp(doc) if not token.is_stop and token.is_alpha])
    return df


def sentiment_analysis(df):
    df['sentiment'] = df['normalized_text'].apply(lambda x: TextBlob(' '.join(x)).sentiment.polarity)
    return df


def calculate_sentiments(df):
    # Exclude zero sentiments
    df_non_zero = df[df['sentiment'] != 0]

    df = df_non_zero.groupby('date').agg(
        average_sentiment=('sentiment', 'mean'),
        sentiment_count=('sentiment', 'count')
    ).reset_index()

    # Convert 'date' column to datetime if it's not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])

    # Create continuous date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start_date, end_date)

    df_dates = pd.DataFrame(dates, columns=['date'])
    df = df_dates.merge(df, on='date', how='left')
    df.fillna(0, inplace=True)

    return df


def add_stock_price(df, query_params):
    # Convert 'date' column to datetime if it's not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])
    start_date = df['date'].min()
    end_date = df['date'].max()
    stock_data = yf.download(query_params['search_term'], start=start_date, end=end_date)
    close_price = stock_data['Close']

    # Merge the close price with the original DataFrame
    df = df.merge(close_price.rename('close_price'), left_on='date', right_index=True, how='left')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['close_price'] = df['close_price'].resample('D').interpolate()
    df = df.reset_index()

    return df


def main():
    create_cache_dir()
    reddit_data_file = os.path.join(cache_directory, 'reddit_data.csv')
    if os.path.isfile(reddit_data_file):
        print("Reading existing reddit data from file ...")
        df = pd.read_csv(reddit_data_file)
        print("Successfully retrieved existing reddit data.")
    else:
        print("\n" + Fore.LIGHTYELLOW_EX + "No existing data file found. Querying reddit..." + Style.RESET_ALL)
        df = query_reddit_data(query_parameters)
        df.to_csv(reddit_data_file, index=False)
        print("Done querying reddit and saved data.")

    if 'normalized_text' not in df.columns:
        print("\nNormalized text not found in data. Proceeding with text normalization...")
        df = process_text(df)
        df.to_csv(reddit_data_file, index=False)
        print("Done with text normalization and updated data.")

    if 'sentiment' not in df.columns:
        print("\nSentiment analysis results not found in data. Proceeding with sentiment analysis...")
        df = sentiment_analysis(df)
        df.to_csv(reddit_data_file, index=False)
        print("Done with sentiment analysis and updated data.")

    dataset_file = os.path.join(cache_directory, 'dataset_file.csv')

    if os.path.isfile(dataset_file):
        print("\nReading existing dataset from file...")
        df = pd.read_csv(dataset_file)
        print("Successfully retrieved existing grouped data.")
    else:
        print("\n" + Fore.LIGHTYELLOW_EX + "No existing dataset found. Proceeding with data aggregation..." + Style.RESET_ALL)

    if 'average_sentiment' not in df.columns or 'sentiment_count' not in df.columns:
        print("\nAverage sentiment and sentiment count not found in data. Proceeding with sentiment calculation...")
        df = calculate_sentiments(df)
        df.to_csv(dataset_file, index=False)
        print("Done with sentiment aggregation and updated data.")

    if 'close_price' not in df.columns:
        print("\nStock close price not found in data. Proceeding with stock price retrieval...")
        df = add_stock_price(df, query_parameters)
        df.to_csv(dataset_file, index=False)
        print("Done with stock price retrieval and updated data.")

    print(Fore.GREEN + "All steps completed successfully." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
