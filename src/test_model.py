import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 1. Charger les données (Assure-toi d'être dans le dossier src)
try:
    X_train = joblib.load('X_train.pkl')
    X_test = joblib.load('X_test.pkl')
    y_train = joblib.load('y_train.pkl')
    y_test = joblib.load('y_test.pkl')
except:
    print("Erreur : Fichiers .pkl introuvables. Lance d'abord preprocess.py !")
    exit()

# 2. Définir les modèles
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# 3. Comparaison
results = []

print("\n--- 🚀 Comparaison des Modèles Heart-Guard ---\n")

for name, model in models.items():
    # Entraînement
    model.fit(X_train, y_train)
    # Prédiction
    y_pred = model.predict(X_test)
    
    # Calcul des scores
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) # Très important en médecine !
    f1 = f1_score(y_test, y_pred)
    
    results.append({"Modèle": name, "Précision (Acc)": acc, "Rappel (Recall)": rec, "F1-Score": f1})

# 4. Affichage du tableau comparatif
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Trouver le gagnant
winner = df_results.loc[df_results['F1-Score'].idxmax()]['Modèle']
print(f"\n🏆 Le meilleur modèle pour ton projet est : {winner}")