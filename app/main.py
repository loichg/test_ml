from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, conint
import joblib
import numpy as np
from pathlib import Path
from typing import List

# Créer l'app
app = FastAPI()

# Charger le modèle
modele = joblib.load("app/rf_depression.pkl")

# Templates HTML
templates = Jinja2Templates(directory="app/templates")

# Schéma d'entrée pour l'API
class InputData(BaseModel):
    features: List[conint(ge=0, le=1)]  # liste de 0/1

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = modele.predict(X)[0]
    proba = modele.predict_proba(X)[0, 1]
    return {"prediction": int(prediction), "proba": float(proba)}

@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(request: Request, features: str = Form(...)):
    try:
        values = list(map(int, features.strip().split(",")))
        X = np.array(values).reshape(1, -1)
        prediction = modele.predict(X)[0]
        proba = modele.predict_proba(X)[0, 1]
        return templates.TemplateResponse("form.html", {
            "request": request,
            "prediction": int(prediction),
            "proba": f"{proba:.2%}",
            "input": features
        })
    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "Erreur dans les données. Entrez une liste comme : 1,0,1,0,1"
        })
