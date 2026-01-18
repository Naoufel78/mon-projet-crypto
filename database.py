from sqlalchemy import create_engine
import pandas as pd

# --- CONFIGURATION DE LA CONNEXION ---
DB_USER = "postgres"
DB_PASSWORD = "allo123allo" 
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "crypto_db"

def get_engine():
    """Crée le moteur de connexion pour PostgreSQL."""
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def save_to_sql(df, table_name):
    """Envoie un tableau (DataFrame) vers la base de données."""
    try:
        engine = get_engine()
        # 'replace' crée la table si elle n'existe pas ou l'écrase si elle existe
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"✅ Succès : Les données sont stockées dans la table '{table_name}' !")
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement : {e}")