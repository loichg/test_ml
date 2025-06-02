from fastapi import FastAPI
from pydantic import BaseModel, conint
import joblib
import numpy as np

modele = joblib.load("app/rf_depression.pkl")

app = FastAPI()

class InputData(BaseModel):
    features: list[conint(ge=0, le=1)]  # liste de 0/1 validée

@app.get("/")
def read_root():
    return {"message": "API opérationnelle"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = modele.predict(X)[0]
    proba = modele.predict_proba(X)[0, 1]
    return {"prediction": int(prediction), "proba": float(proba)}
