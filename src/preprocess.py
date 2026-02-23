import pandas as pd
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE

import joblib


# 1. Chargement (Vérifie bien le chemin !)
# Si tu lances le script depuis le dossier 'src', le chemin est ../data/heart.csv
"""try:
    df = pd.read_csv("../data/heart.csv", sep=';')
    print("✅ Fichier chargé avec succès !")
except FileNotFoundError:
    print("❌ Erreur : Le fichier heart.csv est introuvable. Vérifie le chemin.")

# 2. Encodage des variables catégorielles (One-Hot Encoding)
# C'est plus propre que LabelEncoder pour XGBoost
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X = pd.get_dummies(X, drop_first=True)

# 3. Division Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Équilibrage avec SMOTE (uniquement sur le train)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Analyse terminée.")
print(f"Taille du set d'entraînement : {X_train_res.shape}")
print(f"Distribution des classes : \n{y_train_res.value_counts()}")



# Sauvegarde des sets de données
joblib.dump(X_train_res, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train_res, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')

# Sauvegarde des colonnes (pour que l'App Streamlit sache quoi afficher)
joblib.dump(X_train_res.columns.tolist(), 'model_columns.pkl')

print("✅ Données prétraitées et sauvegardées !")"""


# 1. Chargement
try:
    df = pd.read_csv("../data/heart.csv", sep=';')
    print("✅ Fichier chargé avec succès !")
except FileNotFoundError:
    print("❌ Erreur : Le fichier heart.csv est introuvable.")

# 2. Encodage
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X = pd.get_dummies(X, drop_first=True)

# 3. Division Train/Test
# On garde stratify=y pour maintenir le ratio 60/40 dans les deux sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Analyse
print(f"Analyse terminée.")
print(f"Taille du set d'entraînement : {X_train.shape}")
print(f"Distribution des classes (Réelle) : \n{y_train.value_counts()}")

# --- SAUVEGARDE ---
# Attention : on sauvegarde X_train et y_train (puisqu'il n'y a plus de "_res")
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')

# Sauvegarde des colonnes
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

print("✅ Données (version originale) sauvegardées !")