# Aplikacja Flask do serwowania modelu KNN
from flask import Flask, request
import numpy as np
import joblib

app = Flask("KNN")
model = joblib.load("knn.model")

@app.route("/predict")
def predict_iris():
    try:
        sl = float(request.args.get("sl",0))
        sw = float(request.args.get("sw",0))
        pl = float(request.args.get("pl",0))
        pw = float(request.args.get("pw",0))

        if sl<=0 or sw<=0 or pl<=0 or pw<=0:
            raise ValueError("jakaś wartość jest <=0")

        result = model.predict([ np.array([sl,sw,pl,pw]) ])
        iris = ["setosa","versicolor","virginica"]
        return str(iris[ result[0] ])

    except Exception as exc:
        return str(exc)

@app.route("/")
def hello():
    return "<h1>Predykcja KNN</h1>"

app.run(debug=True, port=1234, host="0.0.0.0")
