import pandas as pd
import numpy as np
import re
from scipy.sparse import data

from scipy.sparse.construct import random
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import train_test_split


class CDataLoader():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer = Tokenizer()
        self.max_review_length = 100
        self.vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', max_features= 10000,strip_accents='unicode', norm='l2')


    def dataframe_loader(self):
        self.df_train =  pd.read_csv(self.train_path)
        self.df_test_main =  pd.read_csv(self.test_path)
        self.df_train = self.df_train.replace({'sentiment' : { 'positive' : 0, 'negative' : 1 }})
        self.df_test_main = self.df_test_main.replace({'sentiment' : { 'positive' : 0, 'negative' : 1 }})
        

    def data_visualiser(self):
        # Use this method to save the Seaborn or matplotlib plots - EDA
        train_vis_plot = sns.countplot(x='sentiment', data=self.df_train)
        test_vis_plot = sns.countplot(x='sentiment', data=self.df_test_main)
        train_vis_plot.savefig("train_distribution_plot.png")
        test_vis_plot.savefig("test_distribution_plot.png")


    def data_process_loader_keras_sequence(self, method =  "train", data_id = ''):
        if method == "train":
            X_total = []
            sentences = list(self.df_train['review'])
            for sen in sentences:
                X_total.append(self.preprocess_text(sen))
            y_total = []
            print(len(X_total))
            y_total = self.df_train['sentiment'].tolist()
            X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.20, random_state= 12)
            
            self.tokenizer.fit_on_texts(X_train)
            X_train = self.text_tokenizer(X_train)
            X_test = self.text_tokenizer(X_test)
            X_train = sequence.pad_sequences(X_train, maxlen= self.max_review_length)
            X_test = sequence.pad_sequences(X_test, maxlen= self.max_review_length)
            return X_train, X_test, y_train, y_test
        elif method == "test":
            X_test = []
            sentences = list(self.df_test_main['review'])
            for sen in sentences:
                X_test.append(self.preprocess_text(sen))
            y_test = self.df_test_main['sentiment'].tolist()
            X_test = self.text_tokenizer(X_test)
            X_test = sequence.pad_sequences(X_test, maxlen = self.max_review_length)
            return X_test, y_test

        elif method == "inference":
            X_infer = []
            sentences  =  list(self.df_test_main.loc[self.df_test_main['review_id']== data_id, 'review'])
            for sen in sentences:
                X_infer.append(self.preprocess_text(sen))
            X_infer = self.text_tokenizer(X_infer)
            X_infer = sequence.pad_sequences(X_infer, maxlen = self.max_review_length)
            return X_infer

        else:
            print("Please check the method parameter - Acceptable Values are train , test and inference")
            pass
  

    def data_process_loader_tf_idf(self, method = "train", data_id = ''):
        if method == "train":
            X_total = []
            sentences = list(self.df_train['review'])
            for sen in sentences:
                X_total.append(self.preprocess_text(sen))
            y_total = []
            y = self.df_train['sentiment'].tolist()
            X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.20, random_state= 12)
            X_train =  self.vectorizer.fit_transform(X_train).todense()
            X_test = self.vectorizer.fit_transform(X_test).todense()
            return X_train, X_test, y_train, y_test

        elif method == "test":
            X_test = []
            sentences = list(self.df_test_main['review'])
            for sen in sentences:
                X_test.append(self.preprocess_text(sen))
            y_test = self.df_test_main['sentiment'].tolist()
            X_test = self.vectorizer.transform(X_test).todense()
            return X_test, y_test

        elif method == "inference":
            X_infer = []
            sentences  =  list(self.df_test_main.loc[self.df_test_main['review_id']== data_id, 'review'])
            for sen in sentences:
                X_infer.append(self.preprocess_text(sen))
            X_infer = self.vectorizer.transform(X_infer).todense()
            return X_infer

        else:
            print("Please check the method parameter - Acceptable Values are train , test and inference")
            pass


    def preprocess_text(self,sen):
        sentence = self.remove_tags(sen)
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    def remove_tags(self, text):
        TAG_RE =  re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    def text_tokenizer(self,data):
        tokenized_data = self.tokenizer.texts_to_sequences(data)
        return tokenized_data




