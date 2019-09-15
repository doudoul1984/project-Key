from flask import Flask, render_template, url_for, request

import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pickle
import json

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    text = request.form['text']
    input_text=text
    nltk.download('wordnet')
    nltk.download('stopwords')
    ###############CLEANING##################
    ##Creating a list of stop words and adding custom stopwords
    stop_words = set(stopwords.words("english"))
    ##Creating a list of custom stopwords
    new_words = ["using", "could", "would", "like", "able", "know", "use", "want", "need", "question"]
    stop_words = stop_words.union(new_words)
    soup = BeautifulSoup(text)
    codetags = soup.find_all('code')
    for codetag in codetags:
        codetag.extract()
    text=soup.get_text()
    #Remove punctuations keeping C#
    text = re.sub('[^a-zA-ZC#]', ' ', text)
    #Convert to lowercase
    text = text.lower()
    #remove tags
    #text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # remove special characters and digits
    #text=re.sub("(\\d|\\W)+"," ",text)
    ##Convert to list from string
    text = text.split()
    ##Stemming
    ps=PorterStemmer()
    #Lemmatisation
    #lem = WordNetLemmatizer()
    #text = [lem.lemmatize(word) for word in text if not word in
    #       stop_words]
    text =[ps.stem(word) for word in text if not word in
            stop_words]
    text = " ".join(text)

   
    return text


if __name__ == "__main__":
    app.run()


