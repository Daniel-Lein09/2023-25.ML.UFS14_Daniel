import logging
import json
import glob
import sys
from os import environ
from flask import Flask
from keras import models
import numpy as np

logging.debug('Init a Flask app')
app = Flask(__name__)


def doit():
    try:
        m =      [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
        
        b = [14.05,27.15,91.38,600.4,0.09929,0.1126,0.04462,0.04304,0.1537,0.06171,0.3645,1.492,2.888,29.84,0.007256,0.02678,0.02071,0.01626,0.0208,0.005304,15.3,33.17,100.2,706.7,0.1241,0.2264,0.1326,0.1048,0.225,0.08321]
        
        model_dir = environ['SM_MODEL_DIR']
        print(f"######## La model dir è: {model_dir}")
        
        model = models.load_model(f"{model_dir}/my_model.keras")
        # model = models.load_model("my_model.keras")
        
        predict_input1 = np.array([
           b
        ])
        predict_result1 = model.predict(predict_input1)

        predict_input2 = np.array([
           m
        ])
        predict_result2 = model.predict(predict_input2)

        
        return json.dumps({
            "predict_result_1": predict_result1.tolist(),
            "predict_result_2": predict_result2.tolist()
        })
    
    except Exception as e:
        return str(e)
    
@app.route('/ping')
def ping():
    logging.debug('Hello from route /ping')

    return doit()
