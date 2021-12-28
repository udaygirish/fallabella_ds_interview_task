import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Conv1D,MaxPooling1D,LSTM,Bidirectional
from tensorflow.keras.layers import Dropout,Flatten,BatchNormalization,LeakyReLU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RNN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import *
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import sequence
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import GlobalMaxPooling1D,MaxPool1D
from tensorflow.keras.layers import Embedding
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


