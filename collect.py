import requests
import pandas as pd

def fetch_crypto_top_10():
    """Récupère le Top 10 des cryptos en direct."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 10,
        'page': 1,
        'sparkline': False
    }
    try:
        print("🌐 Connexion à CoinGecko (Top 10)...")
        response = requests.get(url, params=params)
        df = pd.DataFrame(response.json())
        colonnes = ['name', 'symbol', 'current_price', 'market_cap', 'price_change_percentage_24h']
        return df[colonnes]
    except Exception as e:
        print(f"❌ Erreur Top 10 : {e}")
        return None

def fetch_historical_data(crypto_id='bitcoin', days=365):
    """Récupère l'historique des prix (pour le Machine Learning)."""
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
    try:
        print(f"🌐 Récupération de l'historique pour {crypto_id}...")
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"✅ {len(df)} jours de données récupérés !")
        return df[['date', 'price']]
    except Exception as e:
        print(f"❌ Erreur historique : {e}")
        return None