from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

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
    sommeil_continu: str = Form(...),
    cigarette_continu: str = Form(...),
    sport_continu: str = Form(...),
    imc: str = Form(...),
    temps_travail_continu: str = Form(...)
):
    try:
        # Conversion virgule -> point
        sommeil = float(sommeil_continu.replace(',', '.'))
        cigarette = float(cigarette_continu.replace(',', '.'))
        sport = float(sport_continu.replace(',', '.'))
        imc_val = float(imc.replace(',', '.'))
        travail = float(temps_travail_continu.replace(',', '.'))

        # Application des bornes
        sommeil = max(4, min(sommeil, 9.99))
        cigarette = min(cigarette, 40)
        sport = min(sport, 450)
        imc_val = max(18.5, min(imc_val, 39.9))
        travail = min(travail, 16)

        # Normalisation entre 0 et 1
        sommeil_n = (sommeil - 4) / (9.99 - 4)
        cigarette_n = cigarette / 40
        sport_n = sport / 450
        imc_n = (imc_val - 18.5) / (39.9 - 18.5)
        travail_n = travail / 16

        features = [
            Douleur, Stress, Nutrition, Solitude, Charge, Satisfaction,
            sommeil_n, cigarette_n, sport_n, imc_n, travail_n
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
