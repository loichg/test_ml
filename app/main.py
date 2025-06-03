from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

# Charger le modèle
modele = joblib.load("app/rf_depression.pkl")

# Templates HTML
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    Douleur: int = Form(...),
    Stress: int = Form(...),
    Nutrition: int = Form(...),
    Solitude: int = Form(...),
    Charge: int = Form(...),
    Satisfaction: int = Form(...),
    sommeil_continu: float = Form(...),
    cigarette_continu: float = Form(...),
    sport_continu: float = Form(...),
    imc: float = Form(...),
    temps_travail_continu: float = Form(...)
):
    try:
        features = [
            Douleur, Stress, Nutrition, Solitude, Charge, Satisfaction,
            sommeil_continu, cigarette_continu, sport_continu, imc, temps_travail_continu
        ]
        X = np.array(features).reshape(1, -1)
        prediction = modele.predict(X)[0]
        proba = modele.predict_proba(X)[0, 1]
        return templates.TemplateResponse("form.html", {
            "request": request,
            "prediction": int(prediction),
            "proba": f"{proba:.2%}",
            "inputs": features
        })
    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "Erreur dans les données. Veuillez vérifier les valeurs saisies."
        })
