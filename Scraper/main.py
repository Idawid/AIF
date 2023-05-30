import os
import praw
import spacy
import csv

# Search query
search_query = 'TSLA'   # Use TSLA as the primary search term
time_filter = 'year'    # Filter posts and comments by a specific time range (in this case, one year)
sort_by = 'relevance'   # Sort the search results by relevance
limit = 2               # Limit the search query to X hits
comment_tree_depth = 2  # Limit the navigation of comment tree depth to X

# Cache unique to the search query
cache_path = 'cache'
cache_directory = f"{search_query}_{time_filter}_{sort_by}_limit{limit}_depth{comment_tree_depth}"
cache_directory = os.path.join(cache_path, cache_directory)
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)

# Cache files related to the search query
query_results = os.path.join(cache_directory, 'TSLA_documents.txt')
normalized_query_results = os.path.join(cache_directory, 'TSLA_documents_NORMALIZED.txt')
hyperparam_results = os.path.join(cache_directory, 'hyperparams.csv')


if __name__ == '__main__':
    documents = []
    print("Retrieving query data ...")
    if os.path.isfile(query_results):
        # Retrieve from the file
        with open(query_results, 'r', encoding='utf-8') as f:
            documents = f.readlines()
        print("Done")
    else:
        print('\033[93m' + 'Query data not found!' + '\033[0m')
        print("Querying the reddit ...")
        reddit = praw.Reddit(client_id='CvDuZ97x1r9f0BnG-1_MRg', client_secret='E-y0IxK4zkRHE6ltzt-jjKcUAZi_QQ', user_agent='WSB scrapper (by /u/VultureGamer')
        subreddit = reddit.subreddit('wallstreetbets')

        for post in subreddit.search(search_query, time_filter=time_filter, sort=sort_by, limit=limit):
            documents.append(post.title)
            documents.append(post.selftext)
            post.comments.replace_more(limit=comment_tree_depth)
            for comment in post.comments.list():
                documents.append(comment.body)
        print("Done")
        with open(query_results, 'w', encoding='utf-8') as f:
            f.write('\n'.join(documents))

    texts = []
    print("Retrieving normalized query data ...")
    if os.path.exists(normalized_query_results):
        with open(normalized_query_results, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:
                texts.append(row)
        print("Done")
    if not texts:
        # Process and save normalized texts to file
        print('\033[93m' + 'Normalized query data not found!' + '\033[0m')
        print("Proceeding with normalization ...")
        nlp = spacy.load('en_core_web_sm')
        for document in documents:
            doc = nlp(document)
            text = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            texts.append(text)
        print("Done")

        with open(normalized_query_results, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Normalized text']) # Add header row
            writer.writerows(texts)
            