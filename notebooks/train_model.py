import pandas as pd
import numpy as np


df = pd.read_csv("data/patients_dakar.csv")
df.columns = df.columns.str.strip()


print(df.dtypes)

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

from sklearn.preprocessing import LabelEncoder

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])
df['toux'] = df['toux'].astype(int)
df['fatigue'] = df['fatigue'].astype(int)
df['maux_tete'] = df['maux_tete'].astype(int)
feature_cols = [
    'age',
    'sexe_encoded',
    'temperature',
    'tension_sys',
    'toux',
    'fatigue',
    'maux_tete',
    'frissons',
    'nausee',
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Classes : {list(model.classes_)}")

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


cm = confusion_matrix(y_test, y_pred)

print("Matrice de confusion :")
print(cm)


print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Prediction")
plt.ylabel("Vrai diagnostic")
plt.title("Matrice de confusion - SenSante")

plt.tight_layout()

# Créer le dossier figures s'il n'existe pas
import os
os.makedirs("figures", exist_ok=True)

plt.savefig("figures/confusion_matrix.png")

plt.show()

print("Image sauvegardee dans figures/confusion_matrix.png")

import joblib
import os


os.makedirs("models", exist_ok=True)


joblib.dump(model, "models/model.pkl")


joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")


joblib.dump(feature_cols, "models/feature_cols.pkl")

print("\nModele et encodeurs sauvegardes dans models/")



model_loaded = joblib.load("models/model.pkl")


le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print("\nModele recharge avec succes !")


nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'frissons': True,
    'nausee': False,
    'region': 'Dakar'
}


sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]


features = [[
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    int(nouveau_patient['frissons']),
    int(nouveau_patient['nausee']),
    region_enc
]]


diagnostic = model_loaded.predict(features)[0]


probas = model_loaded.predict_proba(features)[0]

print("\n===== RESULTAT PREDICTION =====")
print(f"Diagnostic predit : {diagnostic}")

print("\nProbabilites :")

for classe, proba in zip(model_loaded.classes_, probas):
    print(f"{classe} : {proba:.2%}")