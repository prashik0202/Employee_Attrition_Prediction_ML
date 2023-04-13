from flask import Flask, render_template, request, redirect, url_for,jsonify, send_file
import pickle
import os
import random
import scipy as sp
import pandas as pd
import numpy as np


app = Flask(__name__)

# Loading machine learning model
model = pickle.load(open('mmm.pkl', 'rb'))

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,27)
    loaded_model = pickle.load(open("mmm.pkl", "rb"))
    predict = loaded_model.predict(to_predict)
    return predict[0]

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template("emp.html")
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        predict = ValuePredictor(to_predict_list)       
        if int(predict)== 0:
            prediction ='Employee Will Not Leave!'
        elif int(predict)<=0.5:
            prediction ='Probability of Employee leaving is Low!'    
        else:
            prediction ='Employee Might Leave!'
               
        return render_template("pred.html", prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)