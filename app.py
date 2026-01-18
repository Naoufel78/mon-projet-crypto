import streamlit as st
import pandas as pd
from database import get_engine

# 1. Configuration de la page
st.set_page_config(page_title="Crypto Dashboard Pro", layout="wide")

st.title("🚀 Dashboard d'Analyse Crypto")

# 2. Fonction pour charger les données
def load_data():
    engine = get_engine()
    query = "SELECT * FROM top_10_cryptos"
    return pd.read_sql(query, engine)

# --- DÉBUT DU BLOC PRINCIPAL ---
try:
    df = load_data()

    # --- INTERFACE UTILISATEUR (FILTRE) ---
    st.sidebar.header("🔍 Filtres")
    liste_cryptos = ["Toutes"] + list(df['name'].unique())
    choix = st.sidebar.selectbox("Choisir une cryptomonnaie :", liste_cryptos)

    if choix != "Toutes":
        df_affiche = df[df['name'] == choix]
    else:
        df_affiche = df

    # --- AFFICHAGE DES CHIFFRES CLÉS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Actifs", len(df_affiche))
    if choix != "Toutes":
        prix = df_affiche['current_price'].values[0]
        variation = df_affiche['price_change_percentage_24h'].values[0]
        col2.metric(f"Prix {choix}", f"${prix:,.2f}", f"{variation:.2f}%")
    
    # --- GRAPHIQUE DE PERFORMANCE ---
    st.subheader("📈 Performance sur 24h (%)")
    df_tri = df.sort_values(by='price_change_percentage_24h', ascending=False)
    st.bar_chart(data=df_tri, x='name', y='price_change_percentage_24h')

    # --- TABLEAU INTERACTIF ---
    st.subheader("📊 Détails du Marché")
    st.dataframe(df_affiche, use_container_width=True)

except Exception as e:
    st.error(f"Erreur de base de données : {e}")

# --- NOUVEAU : AFFICHAGE DE LA PRÉDICTION ML (BIEN ALIGNÉ À GAUCHE) ---
st.divider() 
st.subheader("🤖 Prédiction IA (Machine Learning)")
st.write("Ce graphique montre la tendance calculée par notre modèle de Régression Linéaire.")

# On affiche l'image générée par ml_predict.py
try:
    st.image("prediction_bitcoin.png", use_container_width=True)
except:
    st.info("Lance d'abord 'python ml_predict.py' dans le terminal pour générer la prédiction.")