import streamlit as st
import pandas as pd
import joblib
import os
# On importe la fonction depuis le dossier src
from src.predict import predict_risk 

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Heart-Guard AI",
    page_icon="🏥",
    layout="centered"
)

# --- CHARGEMENT DU MODÈLE POUR L'EXPLICABILITÉ ---
@st.cache_resource
def load_essentials():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, 'src', 'heart_model.pkl'))
    columns = joblib.load(os.path.join(base_dir, 'src', 'model_columns.pkl'))
    return model, columns

model, model_columns = load_essentials()

# --- INTERFACE UTILISATEUR ---
st.title("🏥 Heart-Guard : Aide au Diagnostic")
st.markdown("Analyse du risque cardiaque par **Régression Logistique**.")

st.divider()

# Formulaire de saisie
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Âge du patient", 1, 120, 45)
    sex = st.selectbox("Sexe", ["M", "F"])
    resting_bp = st.number_input("Tension artérielle (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholestérol (mg/dl)", 0, 600, 200)

with col2:
    cp = st.selectbox("Type de douleur thoracique", options=["ASY", "NAP", "ATA", "TA"])
    max_hr = st.number_input("Fréquence cardiaque max", 60, 220, 150)
    oldpeak = st.slider("Dépression ST (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Pente du segment ST", ["Up", "Flat", "Down"])

st.write("") # Espace vide

# --- LOGIQUE DE PRÉDICTION ---
# Le code ci-dessous ne s'exécute QUE lors du clic
if st.button("Lancer l'analyse du risque", use_container_width=True):
    
    # 1. Préparation du dictionnaire pour la fonction
    patient_data = {
        'Age': age, 'Sex': sex, 'RestingBP': resting_bp,
        'Cholesterol': chol, 'ChestPainType': cp,
        'MaxHR': max_hr, 'Oldpeak': oldpeak, 'ST_Slope': slope
    }

    # 2. Appel de la fonction externe
    risk_percent, input_df = predict_risk(patient_data)

    # 3. Affichage des résultats
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Probabilité de Risque", value=f"{risk_percent}%")
        if risk_percent > 50:
            st.error("🚨 RISQUE ÉLEVÉ")
        else:
            st.success("✅ RISQUE FAIBLE")

    with res_col2:
        st.write("### Recommandation")
        if risk_percent > 50:
            st.write("Le modèle suggère une forte probabilité de pathologie. Un examen clinique est conseillé.")
        else:
            st.write("Les indicateurs sont rassurants. Continuez un suivi régulier.")

    # 4. EXPLICABILITÉ
    st.subheader("🔍 Pourquoi ce score ?")
    # Calcul de l'impact : Coeff du modèle * Valeur saisie
    impact = model.coef_[0] * input_df.values[0]
    
    influence_df = pd.DataFrame({
        'Facteur': model_columns,
        'Impact': impact
    }).sort_values(by='Impact', ascending=False)

    st.bar_chart(influence_df.set_index('Facteur'))

st.sidebar.info("Heart-Guard v1.0")