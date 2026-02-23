import shap
import joblib
import matplotlib.pyplot as plt

# 1. Charger le modèle et les données
model = joblib.load('heart_model.pkl')
X_train = joblib.load('X_train.pkl')
X_test  = joblib.load('X_test.pkl')

# 2. Créer l'expliqueur (TreeExplainer est plus rapide pour XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 3. Afficher le Summary Plot
# Ce graphique montre l'importance globale de chaque caractéristique
print("Génération du Summary Plot...")
shap.summary_plot(shap_values, X_test)

# 4. Petit bonus : Le Waterfall Plot pour le PREMIER patient du test
# Cela montre comment on passe de la probabilité de base à la prédiction finale
plt.figure()
shap.plots.bar(explainer(X_test)[0])