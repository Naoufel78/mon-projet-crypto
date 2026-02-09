import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from database import get_engine
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import nltk
from nlp_analysis import get_top_keywords
import sys
import subprocess
from wordcloud import WordCloud  # Nécessite pip install wordcloud

# --- CONFIGURATION NLTK ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords') 

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("🚀 Dashboard & Prédictions Crypto")

# --- 2. FONCTIONS DE CHARGEMENT DES DONNÉES (CACHE) ---
@st.cache_data
def load_top10_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM top_10_cryptos", engine)

@st.cache_data
def load_bitcoin_history():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM historique_bitcoin ORDER BY date ASC", engine)

@st.cache_data
def load_sentiment_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM news_sentiment", engine)

# =========================================================
# BARRE LATÉRALE : ACTUALISATION & FILTRES
# =========================================================
st.sidebar.header("⚙️ Options")

# BOUTON D'ACTUALISATION
if st.sidebar.button("🔄 Actualiser les données"):
    with st.spinner("Récupération des nouvelles données en cours (cela peut prendre quelques secondes)..."):
        # On utilise sys.executable pour être sûr d'utiliser le python du venv
        subprocess.run([sys.executable, "main.py"])
        # On vide le cache pour forcer le rechargement
        st.cache_data.clear()
    
    st.sidebar.success("Données mises à jour !")
    st.rerun()

st.sidebar.divider()

# =========================================================
# PARTIE 1 : LE TABLEAU DE BORD (Top 10 Cryptos)
# =========================================================
st.header("📊 Vue d'ensemble du Marché")

try:
    df = load_top10_data()

    # Filtres Sidebar
    st.sidebar.header("🔍 Filtres")
    liste_cryptos = ["Toutes"] + list(df['name'].unique())
    choix = st.sidebar.selectbox("Choisir une cryptomonnaie :", liste_cryptos)

    if choix != "Toutes":
        df_affiche = df[df['name'] == choix]
    else:
        df_affiche = df

    # Affichage des chiffres clés
    col1, col2, col3 = st.columns(3)
    col1.metric("Actifs suivis", len(df_affiche))
    
    if choix != "Toutes":
        prix = df_affiche['current_price'].values[0]
        variation = df_affiche['price_change_percentage_24h'].values[0]
        col2.metric(f"Prix {choix}", f"${prix:,.2f}", f"{variation:.2f}%")
    
    # Graphique de performance
    st.subheader("Performance sur 24h (%)")
    df_tri = df.sort_values(by='price_change_percentage_24h', ascending=False)
    st.bar_chart(data=df_tri, x='name', y='price_change_percentage_24h')

    # Tableau interactif
    st.subheader("Détails des prix")
    st.dataframe(df_affiche, use_container_width=True)

except Exception as e:
    st.error(f"Erreur lors du chargement du Top 10 : {e}")


st.divider() # Ligne de séparation visuelle


# =========================================================
# PARTIE 2 : ANALYSE DE SENTIMENT (NLP)
# =========================================================
st.header("🌡️ Sentiment du Marché (Analyse NLP)")

try:
    df_news = load_sentiment_data()
    moyenne_sentiment = df_news['sentiment_score'].mean()
    
    col_sent1, col_sent2 = st.columns([1, 3])
    
    with col_sent1:
        if moyenne_sentiment > 0.05:
            st.success(f"POSITIVE ({moyenne_sentiment:.2f})")
        elif moyenne_sentiment < -0.05:
            st.error(f"NÉGATIVE ({moyenne_sentiment:.2f})")
        else:
            st.warning(f"NEUTRE ({moyenne_sentiment:.2f})")

    with col_sent2:
        st.caption("Basé sur les dernières news analysées.")

    with st.expander("Voir les titres analysés"):
        st.dataframe(df_news[['title', 'sentiment_score']], use_container_width=True)

except Exception as e:
    st.info("Pas de données de sentiment disponibles (cliquez sur Actualiser).")


st.divider() # Ligne de séparation visuelle


# =========================================================
# PARTIE 3 : PRÉDICTION AVANCÉE (Machine Learning)
# =========================================================
st.header("🤖 Prédiction de Tendance (Régression Polynomiale)")
st.markdown("Ajustez les paramètres ci-dessous pour voir comment le modèle interprète la courbe.")

try:
    df_btc = load_bitcoin_history()
    # Conversion date
    df_btc['date'] = pd.to_datetime(df_btc['date'])
    df_btc['n_jour'] = np.arange(len(df_btc))

    # --- PARAMÈTRES INTERACTIFS ---
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        degree = st.slider("Complexité de la courbe (Degré)", 1, 10, 4)
    with col_param2:
        days_forecast = st.slider("Jours à prédire", 10, 365, 30)

    # --- CALCUL DU MODÈLE EN DIRECT ---
    X = df_btc[['n_jour']]
    y = df_btc['price']
    
    # Création et entraînement du Pipeline
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    # Prédictions
    y_pred = model.predict(X) # Sur le passé
    
    # Futur
    last_day = df_btc['n_jour'].iloc[-1]
    X_future = np.arange(last_day + 1, last_day + 1 + days_forecast).reshape(-1, 1)
    y_future = model.predict(X_future)
    future_dates = [df_btc['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, days_forecast + 1)]

    score = r2_score(y, y_pred)
    st.caption(f"Précision mathématique du modèle ($R^2$) : **{score:.4f}**")

    # --- GRAPHIQUE ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_btc['date'], y, label="Historique Réel", color='blue', alpha=0.5)
    ax.plot(df_btc['date'], y_pred, label=f"Tendance (Degré {degree})", color='orange', linewidth=2)
    ax.plot(future_dates, y_future, label="Projection Future", color='red', linestyle='--', linewidth=2)
    
    ax.set_title("Projection du Bitcoin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erreur lors de la prédiction ML : {e}")


# =========================================================
# PARTIE 4 : NUAGE DE MOTS (NLP)
# =========================================================
st.divider()
st.subheader("🗣️ De quoi parle le marché ? (WordCloud)")

try:
    keywords_dict = get_top_keywords()
    
    if keywords_dict:
        # Création de deux colonnes pour l'affichage
        col_wc1, col_wc2 = st.columns([2, 1])

        with col_wc1:
            st.markdown("#### Nuage de mots")
            # Création du WordCloud
            wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(keywords_dict)
            
            # Affichage Matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off') # On cache les axes
            st.pyplot(fig)

        with col_wc2:
            st.markdown("#### Top Mots-clés")
            # Tableau simple
            df_keywords = pd.DataFrame(list(keywords_dict.items()), columns=['Mot', 'Freq']).sort_values(by='Freq', ascending=False)
            st.dataframe(df_keywords, hide_index=True, use_container_width=True)

    else:
        st.info("Pas assez de données pour générer le nuage de mots.")

except Exception as e:
    st.error(f"Erreur NLP/WordCloud : {e}")