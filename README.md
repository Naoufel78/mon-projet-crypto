# 🚀 Dashboard d'Analyse et de Prédiction Crypto

## 📝 Présentation du Projet
Ce projet est un pipeline de données complet (ETL) qui récupère les prix des cryptomonnaies, les stocke dans une base SQL et utilise le Machine Learning pour prédire les tendances.

## 🛠️ Technologies Utilisées
- **Langage** : Python 3.13
- **Base de données** : PostgreSQL 17
- **Analyse & ML** : Pandas, Scikit-Learn
- **Visualisation** : Streamlit

## ⚙️ Installation
1. Activer l'environnement virtuel : `source venv/Scripts/activate`
2. Installer les dépendances : `pip install -r requirements.txt`

## 🚀 Utilisation
- **Collecte des données** : `python main.py` (Met à jour le Top 10 et l'historique)
- **Calcul de l'IA** : `python ml_predict.py` (Génère le graphique de prédiction)
- **Lancer le Dashboard** : `streamlit run app.py`