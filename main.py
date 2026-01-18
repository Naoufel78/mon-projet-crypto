from collect import fetch_crypto_top_10, fetch_historical_data
from database import save_to_sql

def run_pipeline():
    print("🚀 Démarrage du pipeline de données...")
    
    # 1. Pour le Dashboard
    df_top10 = fetch_crypto_top_10()
    if df_top10 is not None:
        save_to_sql(df_top10, "top_10_cryptos")
    
    # 2. Pour le Machine Learning
    df_history = fetch_historical_data('bitcoin', 365)
    if df_history is not None:
        save_to_sql(df_history, "historique_bitcoin")
        print("🏁 Pipeline terminé avec succès !")

if __name__ == "__main__":
    run_pipeline()