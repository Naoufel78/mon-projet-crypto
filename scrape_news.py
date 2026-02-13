import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from database import save_to_sql

def analyze_crypto_news():
    print("📰 Récupération des dernières news...")
    # Flux RSS de CoinDesk
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    feed = feedparser.parse(url)
    
    analyzer = SentimentIntensityAnalyzer()
    news_data = []

    # --- PARTIE QUI MANQUAIT ---
    # On boucle sur chaque article (entry) trouvé dans le flux
    # On limite à 10 ou 20 pour ne pas surcharger
    for entry in feed.entries[:100]:
        title = entry.title
        # L'IA analyse le titre
        scores = analyzer.polarity_scores(title)
        compound_score = scores['compound']
        
        # On stocke le titre et son score
        news_data.append({
            'title': title,
            'sentiment_score': compound_score,
            'date': entry.published
        })
    # ---------------------------

    df_news = pd.DataFrame(news_data)
    
    if not df_news.empty:
        print(f"✅ {len(df_news)} news analysées.")
        # On sauvegarde dans la base de données
        save_to_sql(df_news, "news_sentiment")
        return df_news
    
    return None

if __name__ == "__main__":
    analyze_crypto_news()