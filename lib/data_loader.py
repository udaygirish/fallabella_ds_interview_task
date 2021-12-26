import pandas as pd
import numpy as np
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
import matplotlib.pyplot as plt
import seaborn as sns

class CDataLoader():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer = Tokenizer()
        self.vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', max_features= 10000,strip_accents='unicode', norm='l2')


    def dataframe_loader(self):
        self.df_train =  pd.read_csv(self.train_path)
        self.df_test =  pd.read_csv(self.test_path)
        self.df_train = self.df_train.replace({'sentiment' : { 'positive' : 0, 'negative' : 1 }})
        self.df_test = self.df_test.replace({'sentiment' : { 'positive' : 0, 'negative' : 1 }})
        self.tokenizer.fit_on_texts(self.df_train)

    def data_visualiser(self):
        # Use this method to save the Seaborn or matplotlib plots - EDA
        train_vis_plot = sns.countplot(x='sentiment', data=self.df_train)
        test_vis_plot = sns.countplot(x='sentiment', data=self.df_test)
        train_vis_plot.savefig("train_distribution_plot.png")
        test_vis_plot.savefig("test_distribution_plot.png")


    def data_process_loader




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
        tokenized_data = self.tokenizer.texts_to_sequence(data)




