import os
import praw
import spacy
import csv
from datetime import date
from datetime import datetime
from textblob import TextBlob

# Search query
search_query = 'NVDA'   # Use NVDA as the primary search term
time_filter = 'year'    # Filter posts and comments by a specific time range (in this case, one year)
sort_by = 'relevance'   # Sort the search results by relevance
limit = 2               # Limit the search query to X hits
comment_tree_depth = 2  # Limit the navigation of comment tree depth to X
comments_per_post = 20  # Limit the comments under post to X

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
            sentiment = TextBlob(normalized_text).sentiment.polarity
            sentiments.append((normalized_text, float(sentiment), data))

        with open(sentiment_analysis_results, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['normalized_text', 'sentiment', 'date'])  # Write the header row

            for text in sentiments:
                writer.writerow(text)

    # average_sentiments = []
    # for date, sentiments in sentiments.items():
    #     average_sentiment = sum(sentiments) / len(sentiments)
    #     average_sentiments.append([average_sentiment, date])