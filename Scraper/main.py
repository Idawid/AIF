import os

import pandas as pd
import praw
import spacy
from tqdm import tqdm
from colorama import Fore, Style
from textblob import TextBlob
from datetime import datetime

query_parameters = {
    # Client details
    'client_id': 'CvDuZ97x1r9f0BnG-1_MRg',
    'client_secret': 'E-y0IxK4zkRHE6ltzt-jjKcUAZi_QQ',
    'user_agent': 'WSB scrapper (by /u/VultureGamer)',
    # Search parameters
    'subreddit': 'wallstreetbets',
    'search_term': 'SPY',
    'time_range': 'year',
    'sort_order': 'relevance',
    # Query limits
    'post_limit': 2,
    'comment_depth': 3,
    'comment_limit': 3,
}

# Cache unique to the search query
cache_path = 'cache'
cache_directory = f"{query_parameters['search_term']}_{query_parameters['time_range']}_{query_parameters['sort_order']}_" \
                  f"limit{query_parameters['post_limit']}_depth{query_parameters['comment_depth']}"
cache_directory = os.path.join(cache_path, cache_directory)
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)


def query_reddit_data(query_params):
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


def main():
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

    print(Fore.GREEN + "All steps completed successfully." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
