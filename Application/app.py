import csv
import os
import tkinter as tk
from tkinter import simpledialog
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
# Search query
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

search_query = 'SPY'    # Use X as the primary search term
time_filter = 'year'    # Filter posts and comments by a specific time range (in this case, one year)
sort_by = 'relevance'   # Sort the search results by relevance
limit = 1000            # Limit the search query to X hits
comment_tree_depth = 2  # Limit the navigation of comment tree depth to X
comments_per_post = 20  # Limit the comments under post to X

# Cache unique to the search query
cache_path = '../Scraper/cache'
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
dataset_results = os.path.join(cache_directory, 'dataset.csv')


# Load the model
model = load_model('../Scraper/gosling.h5')


combined_data = []

if os.path.exists(dataset_results):
    # Retrieve from the file
    with open(dataset_results, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            tupled_row = tuple(row)  # Convert the row to a tuple
            combined_data.append((float(tupled_row[0]), float(tupled_row[1]), float(tupled_row[2]), tupled_row[3]))

df_combined = pd.DataFrame(combined_data, columns=['Close Price', 'Sentiment Average', 'Sentiment Count', 'Date'])
df_combined = df_combined.set_index('Date')

df = df_combined[['Close Price', 'Sentiment Average', 'Sentiment Count']]

dataset = df.values
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Used in the prediction
last_x_days = scaled_data[-60:]
next_month_prices = []

# App root
root = tk.Tk()
root.geometry('600x600')
root.title("Stock prices prediction using NLP")

title = tk.Label(root, text="Stock prices prediction using NLP", font=("Helvetica", 16))
title.pack()
authors = tk.Label(root, text="Authors: Dawid Mączka, Nikodem Olszowy\nMateusz Sudejko, Maciej Sajecki", font=("Helvetica", 11))
authors.pack()

label = tk.Label(root, text="Stock: SPY | Predict for how many days?", font=("Helvetica", 16))
label.pack()
input_days = tk.Entry(root)
input_days.pack()

fig = None
canvas = None
toolbar = None

def predict():
    global fig
    global canvas
    global toolbar
    global last_x_days
    global next_month_prices

    predict_for_x_days = int(input_days.get())

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

    previous_prices = scaler.inverse_transform(scaled_data[-60:])[:, 0]

    dates = pd.date_range(start='2023-05-31', periods=predict_for_x_days)
    data = {'Dates': dates, 'Prices': next_month_prices}
    df = DataFrame(data)

    # Clear the previous plot before drawing
    if canvas:
        canvas.get_tk_widget().pack_forget()
    if toolbar:
        toolbar.pack_forget()

    # Plot the data
    fig = Figure(figsize=(5, 5), dpi=100)
    plot = fig.add_subplot(1, 1, 1)

    start_date = datetime.now() - timedelta(days=60)
    start_date_str = start_date.strftime('%Y-%m-%d')

    combined_dates = pd.date_range(start=start_date_str, periods=len(previous_prices) + len(next_month_prices))
    plot.plot(combined_dates[:len(previous_prices)], previous_prices, color='black')
    plot.plot(combined_dates[len(previous_prices):], next_month_prices, color='red')

    # format the axis
    # Enlarge the y-axis
    y_min = min(previous_prices.tolist() + next_month_prices)  # Minimum y-value
    y_max = max(previous_prices.tolist() + next_month_prices)  # Maximum y-value
    y_margin = (y_max - y_min) * 0.2  # Add a 10% margin to the y-axis limits
    plot.set_ylim(y_min - 3 * y_margin, y_max + y_margin)
    plot.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Add the plot to the GUI
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    # Add the plot and toolbar to GUI
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    toolbar.pack()

    # Clear
    last_x_days = scaled_data[-60:]
    next_month_prices = []


# print(previous_prices)
# print(next_month_prices)


predict_button = tk.Button(root, text='Predict', command=predict)
predict_button.pack()

# Run the GUI
root.mainloop()