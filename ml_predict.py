import pandas as pd
import matplotlib.pyplot as plt
from database import get_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def run_ml_prediction():
    print("🧠 Chargement des données historiques...")
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM historique_bitcoin ORDER BY date ASC", engine)

    # 1. PRÉPARATION DES DONNÉES (Feature Engineering)
    # On crée une colonne 'n_jour' (1, 2, 3...) car l'IA comprend mieux les nombres que les dates
    df['n_jour'] = np.arange(len(df))
    
    X = df[['n_jour']] # La donnée d'entrée (le temps qui passe)
    y = df['price']    # Ce qu'on veut prédire (le prix)

    # 2. ENTRAÎNEMENT : On garde 80% pour apprendre et 20% pour tester
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("🤖 Entraînement du modèle (Régression Linéaire)...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. PRÉDICTION
    predictions = model.predict(X_test)

    # 4. VISUALISATION
    print("📊 Génération du graphique de prédiction...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], label="Prix Réel", color='blue')
    plt.plot(df['date'].iloc[len(X_train):], predictions, label="Prédiction (Tendance)", color='red', linestyle='--')
    plt.title("Prédiction de la tendance du Bitcoin")
    plt.legend()
    plt.savefig("prediction_bitcoin.png") # Sauvegarde le graphique en image
    print("✅ Graphique sauvegardé sous 'prediction_bitcoin.png'")

if __name__ == "__main__":
    run_ml_prediction()