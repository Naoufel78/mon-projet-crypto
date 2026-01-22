import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from database import save_to_sql

def analyze_crypto_news():
    print("📰 Récupération des dernières news...")
    # Flux RSS de CoinDesk (très stable)
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    feed = feedparser.parse(url)
    
    analyzer = SentimentIntensityAnalyzer()
    news_data = []

    for entry in feed.entries[:10]: # On prend les 10 dernières news
        # L'IA calcule le score : de -1 (très négatif) à +1 (très positif)
        vs = analyzer.polarity_scores(entry.title)
        score = vs['compound']
        
        news_data.append({
            'title': entry.title,
            'date': entry.published,
            'sentiment_score': score
        })

    df_news = pd.DataFrame(news_data)
    
    if not df_news.empty:
        print(f"✅ {len(df_news)} news analysées.")
        save_to_sql(df_news, "news_sentiment")
        return df_news
    return None

if __name__ == "__main__":
    analyze_crypto_news()