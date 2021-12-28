import fastapi
import os, sys

from fastapi import FastAPI, HTTPException
from scipy.sparse import data
from starlette.middleware.cors import CORSMiddleware
import uvicorn

from lib import data_loader, model_creator
import tensorflow as tf
import numpy as np
import time

app = fastapi.FastAPI()

dataloader = data_loader.CDataLoader('./data/reviews_training_26000.csv','./data/reviews_test_4000.csv')
model_create = model_creator.ModelCreator()
model = model_create.load_dl_model("./weights/best_weight/weights_best_cnn3_new_2CNN_final.hdf5")
dataloader.dataframe_loader()

X_train,X_test, y_train, y_test = dataloader.data_process_loader_keras_sequence(method = "train")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers= ["*"]
)

@app.get("/")
def entry_page():
    return "Welcome to the base Page of Sentiment Classification API"

@app.post("/predict_sentiment/")
def predict_sentiment(review_id:  str= "T_0"):
    try:
        t1 = time.time()
        X_infer = dataloader.data_process_loader_keras_sequence(method = "inference", data_id = review_id)
        print(len(X_train), len(X_infer))

        y_pred_output = model.predict(np.array(X_infer))

        y_pred_output = ["negative" if i>0.5 else "positive" for i in y_pred_output]
        t2 = time.time()

        print("Inference_Time:{}".format(t2-t1))
        return y_pred_output[0]
    except Exception as e:
        print("Exception is :{}".format(e))
        return "The review_id does not exist in the test set. Please verify"

if __name__ == "__main__":
    uvicorn.run('main_api:app', host = '0.0.0.0', port = 5005, proxy_headers = True, reload=True)


    

