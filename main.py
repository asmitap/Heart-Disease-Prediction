from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("heart.csv")
pipe = pickle.load(open("logistic_model.pkl", 'rb'))


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    # For rendering results on html GUI
    # takes inputs
    inp_features = [float(x) for x in request.form.values()]
    # convert input values to array
    final_features = [np.array(inp_features)]
    predictions = pipe.predict(final_features)
    # return render_template('index.html', predictions=predictions)
    if predictions == 1:
        return render_template('index.html', predictions="The person has Heart Disease")
    else:
        return render_template('index.html', predictions="The person does not have a Heart Disease")


if __name__ == '__main__':
    app.run(debug=True)