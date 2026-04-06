import pandas as pd

df = pd.read_csv("data/patients_dakar.csv")

print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

print("\n--- 5 premiers patients ---")
print(df.head())

print("\n--- Statistiques descriptives ---")
print(df.describe().round(2))

print("\n--- Répartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()

for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"{diag} : {count} patients ({pct:.1f}%)")

print("\n--- Répartition par région ---")
print(df["region"].value_counts().head())

print("\n--- Température moyenne par diagnostic ---")
print(df.groupby("diagnostic")["temperature"].mean())

# EXERCICE 1
print("\n--- Patients par sexe et diagnostic ---")
print(df.groupby(["sexe", "diagnostic"]).size())