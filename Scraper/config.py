import os

query_parameters = {
    # Client details
    'client_id': 'CvDuZ97x1r9f0BnG-1_MRg',
    'client_secret': 'E-y0IxK4zkRHE6ltzt-jjKcUAZi_QQ',
    'user_agent': 'WSB scrapper (by /u/VultureGamer)',
    # Search parameters
    'subreddit': 'wallstreetbets',
    'search_term': 'NVDA',
    'time_range': 'year',
    'sort_order': 'relevance',
    # Query limits
    'post_limit': 2000,
    'comment_depth': 3,
    'comment_limit': 100,
}

# Cache unique to the search query
cache_path = 'cache'
cache_directory = f"{query_parameters['search_term']}_{query_parameters['time_range']}_{query_parameters['sort_order']}_" \
                  f"limit{query_parameters['post_limit']}_depth{query_parameters['comment_depth']}"
cache_directory = os.path.join(cache_path, cache_directory)

models_path = 'models'


def create_cache_dir():
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)