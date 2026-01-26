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



    df_news = pd.DataFrame(news_data)
    
    if not df_news.empty:
        print(f"✅ {len(df_news)} news analysées.")
        save_to_sql(df_news, "news_sentiment")
        return df_news
    return None

if __name__ == "__main__":
    analyze_crypto_news()