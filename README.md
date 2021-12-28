# fallabella_ds_interview_task
Interview Task Submission for Fallabella Data Scientist Postion

#### Please follow the below link to access Weights & Biases Dashboard
https://wandb.ai/udaygirish/fallabella_sentiment_classification?workspace=user-udaygirish

#### Current Accuracy Metrics
1. Current Best Model Weights -
    Train - Around 98 percent precision
    Test - Around 83 percent 
    Test Confusion Matrix Values - [ [1652,367], [303,1678]]


#### Current Code also support Train using custom datasets and custom models can be defined in Model_Creator.py
#### Current code supports logging checkpoint and callbacks in Weights and biases.

#### API Endpoint 
 
Please hit the API endpoint at the given address and go to /docs address to test it.
Total path: http://<external_ip>:5005/docs 
            Check predict_sentiment
            Otherwise use post request with the review_id on http://<external_ip>:5005/predict_sentiment 



