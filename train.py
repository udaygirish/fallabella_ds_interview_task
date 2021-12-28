import argparse

from tensorflow.python.keras.callbacks import ModelCheckpoint
from lib import data_loader, model_creator
import os, sys
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from matplotlib import pyplot as plt



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-trp", "--train_path", required=False, default="./data/reviews_training_26000.csv",
                        help = "Set training path to the necessary location")

    parser.add_argument("-tep", "--test_path", required=False, default="./data/reviews_test_4000.csv",
                        help = "Set test path to the necessary location")

    parser.add_argument("-e","--epochs", default=20, required=False,
                        help = "Number of epochs to be trained on")

    parser.add_argument("-wp","--weights_path", default = "./weights/best_weight/weights_best_cnn3_new_2CNN_test.hdf5",
                        help = "Path to the Weights (Currently supports only hdf5")

    parser.add_argument("-wt", "--wandb_token", required=True, default=None, 
                        help = "Wandb token - Unique Id to log the model metrics and performance")

    parser.add_argument("-lr", "--learning_rate", required = False, default= 0.0001, 
                        help = "Set learning rate to train the model")

    parser.add_argument("-ld", "--learning_decay", required=False, default=1e-6, 
                        help = "Set learning decay to enable it in the optimizer")

    args = parser.parse_args()

    dataloader = data_loader.CDataLoader(args["train_path"], args["test_path"])

    model_create = model_creator.ModelCreator()


    # Load the dataframes
    dataloader.dataframe_loader()

    # Uncomment this if you want to visualise the distributions
    dataloader.data_visualiser()

    X_train, X_test, y_train, y_test = dataloader.data_process_loader_keras_sequence(method = "train")

    X_test_main, y_test_main = dataloader.data_process_loader_keras_sequence(method = "test")

    os.system("wandb login "+str(args["wandb_token"]))
    # Initiate Wandb Logging
    wandb.init(project = "fallabella_sentiment_classification")

    # Model create and fit

    model = model_create.create_dl_model()

    adam_optimizer = tf.keras.optimizers.Adam(lr= args["learning_rate"],decay = args["learning_decay"], clipnorm =1.)
    model.compile(loss= "binary_crossentropy", optimizer = adam_optimizer, metrics = [tf.keras.metrics.Accuracy, tf.keras.metrics.Precision, tf.keras.metrics.Recall])
    print(model.summary())
    filepath = args["weights_path"]
    checkpoint = ModelCheckpoint(filepath, monitor = "val_precision", verbose=1, save_best_only=True, mode = 'max', save_weights_only=False)
    callbacks_list = [checkpoint, WandbCallback()]
    H =  model.fit(X_train, y_train, epochs = args["epochs"], batch_size= 128,verbose=1, callbacks = callbacks_list, validation_data=(X_test,y_test))

    y_main_pred = model.predict(np.array(X_test_main))

    y_main_pred = [1 if i>0.5 else 0 for i in y_main_pred]

    test_accuracy = accuracy_score(y_main_pred, y_test_main)

    print("The test accuracy is:{}".format(test_accuracy))

    confusion_matrix_value = confusion_matrix(y_test_main, y_main_pred, labels=[0,1])

    print ("Confusion matrix:{}".format(confusion_matrix_value))

    # Modify Confusion matrix method
    # disp = ConfusionMatrixDisplay(confusion_matrix= confusion_matrix_value, display_labels = ['positive', 'negative'])
    # disp.plot()
    # plt.savefig('confusion_matrix.png')


    

    
