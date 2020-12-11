from flask import Flask, request, jsonify, render_template
import traceback 
import torch 
import joblib
import sys 
import pandas as pd
from network import Network
import time
import numpy as np

app = Flask(__name__)

@app.before_first_request


def load_model_to_app():
    MODEL = Network()
    MODEL.load_state_dict(torch.load('./static/model/model.bin', map_location=torch.device('cpu')))
    MODEL.to('cpu')
    MODEL.eval()
    app.predictor = MODEL

@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():
    classes =("high","mid","low")
    #Get the data from the form
    data = [request.form['fixedacidity'],
    request.form['volatile_acidity'],
    request.form['citric_acid'],
    request.form['residual_sugar'],
    request.form['chloride'],
    request.form['free_sulfur_dioxide'],
    request.form['total_sulfur_dioxide'],
    request.form['density'],
    request.form['pH'],
    request.form['sulphates'],
    request.form['alcohol']]
    data = np.asarray(data)
    data = data.reshape(1,-1)      
    scaler = joblib.load("../output/scaler.pkl")
    xfeat = scaler.transform(data)
    features = torch.tensor(xfeat).float().unsqueeze(0)      
    predictions = app.predictor(features)
    predict = predictions.squeeze(1)
    predict = predict.argmax(dim=1)
    predict =classes[predict]

    print('INFO Predictions: {}'.format(predictions))


    #class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]

    return render_template('index.html', pred=predict)


if __name__=="__main__":
    app.run(debug=True)