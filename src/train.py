import pandas as pd
#import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np




# --- REPRISE DU PREPROCESS ---
df = pd.read_csv("../data/heart.csv", sep=';')
X = pd.get_dummies(df.drop("HeartDisease", axis=1), drop_first=True)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# --- ENTRAÎNEMENT DU MODÈLE ---
print("🚀 Entraînement du modèle Régression Logistique ...")
"""model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_res, y_train_res)"""
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- EXPLICABILITÉ AVEC SHAP ---
print("🧠 Analyse de l'explicabilité (SHAP)...")


"""explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# Affichage du graphique récapitulatif
shap.summary_plot(shap_values, X_test)"""

# Récupérer les coefficients
importances = model.coef_[0]
feature_names = X.columns

# Créer un DataFrame pour l'affichage
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Affichage du graphique
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Poids du coefficient (Influence)')
plt.title('Importance des facteurs sur le risque cardiaque')
plt.gca().invert_yaxis() # Pour avoir les plus importants en haut
plt.tight_layout()
plt.show()


# On ne sauvegarde QUE le modèle ici
joblib.dump(model, 'heart_model.pkl')
print("✅ Modèle entraîné et sauvegardé !")



# --- PHASE DE TEST SUR UN PATIENT ---
print("\n--- Test de prédiction ---")

# On prend un patient au hasard dans le set de test
sample_patient = X_test.iloc[0:1] 

# On récupère les probabilités [Sain, Malade]
prediction_proba = model.predict_proba(sample_patient)
prediction_label = model.predict(sample_patient)

risk_percent = prediction_proba[0][1] * 100

print(f"Résultat : {'⚠️ Risque de Maladie' if prediction_label[0] == 1 else '✅ Sain'}")
print(f"Probabilité précise : {risk_percent:.2f}%")

# Optionnel : Voir ce qui a causé cette décision pour CE patient précis
"""shap_values_patient = explainer.shap_values(sample_patient)
shap.force_plot(explainer.expected_value, shap_values_patient, sample_patient, matplotlib=True)"""