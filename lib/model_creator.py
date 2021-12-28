import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense,Embedding,Conv1D,MaxPooling1D,LSTM,Bidirectional
from tf.keras.layers import Dropout,Flatten,BatchNormalization,LeakyReLU
from tf.keras.layers import LSTM
from tf.keras.layers import RNN
from tf.keras.callbacks import ModelCheckpoint
from tf.keras.utils import np_utils
from tf.keras import *
import tf.keras as keras
from tf.keras.preprocessing import sequence
from nltk.corpus import stopwords
from tf.keras.preprocessing.text import one_hot
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.layers.core import Activation, Dropout, Dense
from tf.keras.layers import GlobalMaxPooling1D,MaxPool1D
from tf.keras.layers.embeddings import Embedding
from sklearn.linear_model import LogisticRegression
import pickle


class ModelCreator():
    
    def __init__(self, top_words=100000, embedding_vector_length = 32, max_review_length=100):
        self.description = "Model Class creator and Pretrained weights loader"
        self.top_words = top_words
        self.embedding_vector_length = embedding_vector_length
        self.max_review_length  = max_review_length

    def create_dl_model(self):
        model = Sequential()
        model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.max_review_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
        model.add(LeakyReLU(alpha=0.001))
        model.add(BatchNormalization())
        # model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
        # model.add(LeakyReLU(alpha=0.001))
        # model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2,padding='same'))
        model.add(Dropout(0.4))
        model.add(Bidirectional(LSTM(128,)))  
        model.add(BatchNormalization())
        #model.add(Dropout(0.3))
        model.add(Dense(256, activation='leaky_relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def create_lr_classifier(self):
        model = LogisticRegression()
        return model

    def load_dl_model(self, model_path):
        model = self.create_dl_model()
        model.load_weights(model_path)
        return model

    def load_lr_model(self,model_path):
        model = pickle.load(open(model_path , 'rb'))
        return model


