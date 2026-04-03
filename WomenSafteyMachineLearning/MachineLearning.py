import tkinter as tk
from tkinter import filedialog
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords

# Main GUI window
main = tk.Tk()
main.title("Analysis of Human Safety in Indian Cities Using Machine Learning on Data")
main.geometry("1300x1200")
main.config(bg='brown')

filename = ""
tweets_list = []
clean_list = []
pos = neu = neg = 0

# Tweet cleaning function
def tweetCleaning(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)

# Upload function
def upload():
    global filename
    filename = filedialog.askopenfilename(
        initialdir="dataset",
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv")]
    )
    print("Selected file:", filename)
    if filename:
        pathlabel.config(text=filename)
        text.delete('1.0', tk.END)
        text.insert(tk.END, filename + " loaded\n")
    else:
        text.insert(tk.END, "No file selected.\n")

# Read CSV tweets
def read():
    text.delete('1.0', tk.END)
    tweets_list.clear()
    try:
        train = pd.read_csv(filename, encoding='iso-8859-1')
        for i in range(len(train)):
            tweet = train.at[i, 'Text']
            tweets_list.append(tweet)
            text.insert(tk.END, tweet + "\n")
        text.insert(tk.END, f"\n\nTotal data found in dataset: {len(tweets_list)}\n\n\n")
    except Exception as e:
        text.insert(tk.END, f"Error reading file: {e}\n")

# Clean tweets
def clean():
    text.delete('1.0', tk.END)
    clean_list.clear()
    for tweet in tweets_list:
        tweet = tweet.strip().lower()
        cleaned = tweetCleaning(tweet)
        clean_list.append(cleaned)
        text.insert(tk.END, cleaned + "\n")
    text.insert(tk.END, f"\n\nTotal cleaned data: {len(clean_list)}\n\n\n")

# Run sentiment analysis
def machineLearning():
    global pos, neu, neg
    text.delete('1.0', tk.END)
    pos = neu = neg = 0
    for tweet in clean_list:
        blob = TextBlob(tweet)
        polarity = blob.polarity
        if polarity <= 0.2:
            neg += 1
            sentiment = "NEGATIVE"
        elif 0.2 < polarity <= 0.5:
            neu += 1
            sentiment = "NEUTRAL"
        else:
            pos += 1
            sentiment = "POSITIVE"

        text.insert(tk.END, f"{tweet}\nPredicted Sentiment: {sentiment}\nPolarity Score: {polarity}\n")
        text.insert(tk.END, "=" * 80 + "\n")

# Display graph
def graph():
    text.delete('1.0', tk.END)
    total = len(clean_list)
    if total == 0:
        text.insert(tk.END, "No data to show graph.\n")
        return

    text.insert(tk.END, f"Safety Factor\n\n")
    text.insert(tk.END, f"Positive : {pos}\n")
    text.insert(tk.END, f"Negative : {neg}\n")
    text.insert(tk.END, f"Neutral  : {neu}\n\n")
    text.insert(tk.END, f"Total data: {total}\n")
    text.insert(tk.END, f"Positive : {pos / total:.2%}\n")
    text.insert(tk.END, f"Negative : {neg / total:.2%}\n")
    text.insert(tk.END, f"Neutral  : {neu / total:.2%}\n")

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos, neg, neu]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Human Safety & Sentiment Graph')
    plt.axis('equal')
    plt.show()

# GUI Layout
font_title = ('times', 16, 'bold')
font_label = ('times', 14, 'bold')
font_text = ('times', 12, 'bold')

title = tk.Label(main, text='Analysis of Human Safety in Indian Cities Using Machine Learning on Data', bg='brown', fg='white', font=font_title, height=3, width=120)
title.place(x=0, y=5)

uploadButton = tk.Button(main, text="Upload Dataset", command=upload, font=font_label)
uploadButton.place(x=50, y=100)

pathlabel = tk.Label(main, bg='brown', fg='white', font=font_label)
pathlabel.place(x=370, y=100)

readButton = tk.Button(main, text="Read data", command=read, font=font_label)
readButton.place(x=50, y=150)

cleanButton = tk.Button(main, text="data Cleaning", command=clean, font=font_label)
cleanButton.place(x=210, y=150)

mlButton = tk.Button(main, text="Run Machine Learning Algorithm", command=machineLearning, font=font_label)
mlButton.place(x=400, y=150)

graphButton = tk.Button(main, text="Human Safety Graph", command=graph, font=font_label)
graphButton.place(x=730, y=150)

text = tk.Text(main, height=25, width=150, font=font_text)
text.place(x=10, y=200)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)

main.mainloop()
