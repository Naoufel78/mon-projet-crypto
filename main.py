from collect import fetch_crypto_top_10, fetch_historical_data
from scrape_news import analyze_crypto_news
from database import save_to_sql

def run_pipeline():
    print("🚀 Démarrage du pipeline global...")
    
    # 1. Collecte Top 10
    df_top10 = fetch_crypto_top_10()
    if df_top10 is not None:
        save_to_sql(df_top10, "top_10_cryptos")
    
    # 2. Historique pour le Machine Learning
    df_history = fetch_historical_data('bitcoin', 365)
    if df_history is not None:
        save_to_sql(df_history, "historique_bitcoin")
        
    # 3. Collecte et Analyse des News (NLP)
    analyze_crypto_news()
    
    print("🏁 Pipeline terminé avec succès !")

if __name__ == "__main__":
    run_pipeline()