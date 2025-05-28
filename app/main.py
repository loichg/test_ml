# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle
modele = joblib.load("app/rf_depression.pkl")

app = FastAPI()

class InputData(BaseModel):
    features: list  # liste de 0/1

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = modele.predict(X)[0]
    proba = modele.predict_proba(X)[0, 1]
    return {"prediction": int(prediction), "proba": float(proba)}
