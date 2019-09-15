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

    ###############KEYWORDS############################
    from sklearn.feature_extraction.text import CountVectorizer

    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
    with open('/home/doudoul/mysite/APP/static/corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    X=cv.fit_transform(corpus)
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    # get feature names
    feature_names=cv.get_feature_names()
    doc=text
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    #Function for sorting tf_idf in descending order
    from scipy.sparse import coo_matrix
    def sort_coo(coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        #use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:

            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]

        return results
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,3)


    return json.dumps(keywords)


if __name__ == "__main__":
    app.run()


