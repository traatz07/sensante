from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware



model = joblib.load("models/model.pkl")
encoder_sexe = joblib.load("models/encoder_sexe.pkl")
encoder_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")


# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au demarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")


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

class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modele LLM utilise"
    )

SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu expliques les resultats medicaux en francais simple
avec quelques mots wolof faciles (30% de mots en wolof et 70% en francais).

Utilise parfois des expressions comme :
- "nak"
- "ndank ndank"
- "Salam Aleykoum" (qui signifie "bonjour" en arabe, tres utilise au Senegal)
- "bul tiit" (qui signifie "n'aies crainte")
- "waaye"

Sois rassurant.
Maximum 3 phrases.
Ne fais jamais de nouveau diagnostic.
Tu expliques uniquement le resultat fourni."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""

    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )

    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=1.0
        )

        explication = response.choices[0].message.content

        return ExplainOutput(
            explication=explication,
            modele_llm="llama-3.1-8b-instant"
        )

    except Exception as e:
        return ExplainOutput(
            explication=f"Erreur lors de l'appel au LLM : {str(e)}",
            modele_llm="erreur"
        )
    return ExplainOutput(explication=explication)

