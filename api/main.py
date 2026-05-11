from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware



model = joblib.load("models/model.pkl")

encoder_sexe = joblib.load("models/encoder_sexe.pkl")
encoder_region = joblib.load("models/encoder_region.pkl")

feature_cols = joblib.load("models/feature_cols.pkl")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API SenSante active"}



class PatientData(BaseModel):
    age: int
    sexe: str
    temperature: float
    tension_sys: int
    toux: bool
    fatigue: bool
    maux_tete: bool
    frissons: bool
    nausee: bool
    region: str




@app.post("/predict")
def predict(data: PatientData):

    
    sexe_encoded = encoder_sexe.transform([data.sexe])[0]
    region_encoded = encoder_region.transform([data.region])[0]

    
    features = [[
        data.age,
        sexe_encoded,
        data.temperature,
        data.tension_sys,
        int(data.toux),
        int(data.fatigue),
        int(data.maux_tete),
        int(data.frissons),
        int(data.nausee),
        region_encoded
    ]]


 
    prediction = model.predict(features)[0]

    
    probabilities = model.predict_proba(features)[0]



    resultats = {}

    for classe, proba in zip(model.classes_, probabilities):
        resultats[classe] = round(float(proba), 4)

    return {
        "diagnostic": prediction,
        "probabilites": resultats
    }
