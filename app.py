import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================
st.set_page_config(
    page_title="Credit Scoring Pro",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# STYLE CSS (DARK MODE)
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
}

.header {
    background: linear-gradient(90deg, #1f2c56, #3b5998);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.card {
    background-color: #020617;
    color: #e5e7eb;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.6);
    margin-bottom: 20px;
}

.result-ok {
    border-left: 8px solid #22c55e;
    background-color: rgba(34,197,94,0.08);
}

.result-bad {
    border-left: 8px solid #ef4444;
    background-color: rgba(239,68,68,0.08);
}

.stButton>button {
    background: linear-gradient(90deg, #1f2c56, #3b5998);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

div[data-testid="stProgress"] > div > div {
    background-color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header">
    <h1>🏦 Credit Scoring Intelligence</h1>
    <p>Analyse du risque de défaut client par Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================
import cloudpickle

with open("credit_scoring_model_cloud.pkl", "rb") as f:
    model = cloudpickle.load(f)
# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Paramètres")
st.sidebar.info("Saisissez les informations client pour évaluer le risque.")

# =========================================================
# FORMULAIRE (10 VARIABLES EXACTES)
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Profil client")
    age = st.number_input("Âge", 18, 100, 35)
    genre = st.selectbox("Genre", ["Homme", "Femme"])
    situation_matrimoniale = st.selectbox(
        "Situation matrimoniale",
        ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf(ve)"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏠 Situation personnelle")
    type_logement = st.selectbox(
        "Type de logement",
        ["Propriétaire", "Locataire", "Logement familial"]
    )
    type_carte = st.selectbox(
        "Type de carte bancaire",
        ["Débit", "Crédit", "Prépayée"]
    )
    flag_liste_noire = st.selectbox(
        "Client sur liste noire",
        ["Non", "Oui"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💼 Situation financière & emploi")
    ratio_endettement = st.slider("Ratio d’endettement", 0.0, 5.0, 0.5)
    revenu_disponible = st.number_input("Revenu disponible (FCFA)", 0, 5_000_000, 150_000)
    anciennete_client = st.number_input("Ancienneté client (mois)", 0, 600, 36)
    anciennete_emploi = st.number_input("Ancienneté emploi (mois)", 0, 600, 24)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PRÉDICTION
# =========================================================
st.markdown("###")

if st.button("🚀 Lancer l'analyse", use_container_width=True):

    input_data = pd.DataFrame([{
        "RATIO_ENDETTEMENT": ratio_endettement,
        "ANCIENNETE_CLIENT_MOIS": anciennete_client,
        "ANCIENNETE_EMPLOI_MOIS": anciennete_emploi,
        "AGE": age,
        "GENRE": genre,
        "TYPE_LOGEMENT": type_logement,
        "TYPE_CARTE": type_carte,
        "SITUATION_MATRIMONIALE": situation_matrimoniale,
        "FLAG_LISTE_NOIRE": 1 if flag_liste_noire == "Oui" else 0,
        "REVENU_DISPONIBLE_FCFA": revenu_disponible
    }])

    prob_default = model.predict_proba(input_data)[0][1]
    score = int((1 - prob_default) * 1000)

    k1, k2, k3 = st.columns(3)
    k1.metric("📉 Risque", f"{prob_default:.2%}")
    k2.metric("💯 Score", f"{score}/1000")
    k3.metric("📊 Statut", "Faible" if prob_default < 0.3 else "Élevé")

    if prob_default < 0.3:
        decision = "✅ Crédit Accordé"
        style = "result-ok"
        explanation = "Client fiable avec faible probabilité de défaut."
    elif prob_default < 0.5:
        decision = "⚠️ Crédit Sous Surveillance"
        style = "result-ok"
        explanation = "Risque modéré. Analyse complémentaire recommandée."
    else:
        decision = "❌ Crédit Refusé"
        style = "result-bad"
        explanation = "Probabilité élevée de défaut."

    st.markdown(f"""
    <div class="card {style}">
        <h2>{decision}</h2>
        <p><b>Probabilité de défaut :</b> {prob_default:.2%}</p>
        <p><b>Score :</b> {score}/1000</p>
        <p>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(score / 1000)