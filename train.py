import argparse
from lib import data_loader, model_creator
import os, sys
import wandb
from wandb.keras import WandbCallback



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






    

    
