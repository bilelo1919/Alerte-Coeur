import joblib
import pandas as pd
import os

# Utilisation de chemins robustes pour éviter les erreurs de dossier
base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, 'heart_model.pkl'))
model_columns = joblib.load(os.path.join(base_dir, 'model_columns.pkl'))

def predict_risk(input_data: dict):
    """
    Prend un dictionnaire, l'aligne sur le modèle et 
    retourne (pourcentage, dataframe_formatté).
    """
    # 1. Conversion en DataFrame
    df_raw = pd.DataFrame([input_data])
    
    # 2. Alignement des colonnes (One-Hot Encoding automatique)
    # reindex va créer les colonnes Sex_M, Sex_F, etc., et mettre 0 si absentes
    df_input = df_raw.reindex(columns=model_columns, fill_value=0)
    
    # 3. Calcul de la probabilité
    proba = model.predict_proba(df_input)[0][1]
    risk_percent = round(proba * 100, 2)
    
    # ON RETOURNE LES DEUX : le score pour le texte, le DF pour le graphique
    return risk_percent, df_input