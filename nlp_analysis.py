import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from database import get_engine

# --- CONFIGURATION NLTK ---
# On s'assure que les dictionnaires sont téléchargés
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

def get_top_keywords():
    """
    Récupère les titres des news, nettoie le texte (NLP)
    et renvoie les mots les plus fréquents.
    """
    engine = get_engine()
    # On récupère les titres des news depuis ta base de données
    df = pd.read_sql("SELECT title FROM news_sentiment", engine)
    
    if df.empty:
        return {}

    # --- 1. PRÉPARATION (Stop Words) ---
    # On charge la liste des mots "inutiles" en anglais (the, is, at, which...)
    stop_words = set(stopwords.words('english'))
    # On ajoute des mots spécifiques aux news crypto qui reviennent tout le temps mais n'apportent pas d'info
    stop_words.update([
        "coindesk", "market", "price", "crypto", "bitcoin", "btc", "ethereum", "eth", 
        "analysis", "price analysis", "market wrap", "daily", "weekly"
    ]) 

    all_tokens = []

    # --- 2. TOKENISATION & NETTOYAGE ---
    for title in df['title']:
        # Tokenization : On coupe la phrase en mots
        tokens = word_tokenize(str(title).lower())
        
        # Nettoyage : On garde uniquement les mots (pas de ponctuation) qui ne sont pas des stop words
        cleaned_tokens = [
            word for word in tokens 
            if word.isalnum() and word not in stop_words and len(word) > 2
        ]
        all_tokens.extend(cleaned_tokens)

    # --- 3. COMPTAGE (Bag of Words) ---
    # On compte combien de fois chaque mot apparaît
    word_counts = Counter(all_tokens)
    
    # On retourne les 20 mots les plus fréquents (pour nourrir le nuage de mots)
    return dict(word_counts.most_common(20))