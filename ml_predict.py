import pandas as pd
import matplotlib.pyplot as plt
from database import get_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline           
import numpy as np

def run_polynomial_prediction():
    print("🚀 Chargement des données...")
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM historique_bitcoin ORDER BY date ASC", engine)

    # 1. PRÉPARATION
    df['n_jour'] = np.arange(len(df))
    X = df[['n_jour']]
    y = df['price']

    # 2. SÉPARATION
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("🤖 Entraînement du modèle (Polynomial)...")
    
    # --- CHANGEMENT MAJEUR ICI ---
    # Modèle de régression polynomiale
    # Étape 1 : On élève les données à la puissance 4 (degré 4) pour créer des courbes.
    # Étape 2 : On applique la régression linéaire sur ces courbes.
    degree = 4 
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # -----------------------------

    model.fit(X_train, y_train)

    # 3. PRÉDICTION
    predictions = model.predict(X_test)

    # 4. VISUALISATION
    print("📊 Génération du graphique...")
    plt.figure(figsize=(12, 6))
    
    # Données réelles
    plt.plot(df['date'], df['price'], label="Prix Réel", color='blue', alpha=0.6)
    
    # Pour que le dessin de la courbe soit joli, on prédit sur TOUTE la période (passé + futur)
    all_predictions = model.predict(X)
    plt.plot(df['date'], all_predictions, label=f"Tendance Polynomiale (Degré {degree})", color='red', linewidth=2)
    
    # On ajoute une ligne verticale pour séparer le passé (train) du futur (test)
    plt.axvline(x=df['date'].iloc[len(X_train)], color='green', linestyle='--', label="Début des prédictions test")

    plt.title(f"Modélisation du Bitcoin - Régression Polynomiale (Degré {degree})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("prediction_poly_bitcoin.png")
    print("✅ Graphique sauvegardé sous 'prediction_poly_bitcoin.png'")

if __name__ == "__main__":
    run_polynomial_prediction()