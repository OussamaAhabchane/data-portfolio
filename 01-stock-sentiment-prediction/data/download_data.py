"""
Script de téléchargement des données financières et de sentiment
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
TICKERS = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
DATA_DIR = Path(__file__).parent

def download_stock_data():
    """Télécharge les données de prix pour les tickers spécifiés"""
    print("📊 Téléchargement des données boursières...")
    
    all_data = {}
    
    for ticker in TICKERS:
        print(f"  • Téléchargement de {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=START_DATE, end=END_DATE)
            
            if not df.empty:
                df['Ticker'] = ticker
                all_data[ticker] = df
                print(f"    ✓ {len(df)} jours téléchargés")
            else:
                print(f"    ✗ Aucune donnée disponible")
        except Exception as e:
            print(f"    ✗ Erreur: {e}")
    
    # Combiner toutes les données
    combined_df = pd.concat(all_data.values(), ignore_index=False)
    
    # Sauvegarder
    raw_dir = DATA_DIR / 'raw'
    raw_dir.mkdir(exist_ok=True)
    
    output_file = raw_dir / 'stock_prices.csv'
    combined_df.to_csv(output_file)
    print(f"\n✓ Données sauvegardées dans {output_file}")
    print(f"  Total: {len(combined_df)} lignes")
    
    return combined_df

def generate_synthetic_sentiment():
    """
    Génère des données de sentiment synthétiques pour la démonstration.
    Dans un vrai projet, ces données viendraient d'APIs (Twitter, News, etc.)
    """
    print("\n📰 Génération de données de sentiment synthétiques...")
    
    # Charger les données de prix pour avoir les dates
    raw_dir = DATA_DIR / 'raw'
    prices_df = pd.read_csv(raw_dir / 'stock_prices.csv', index_col=0, parse_dates=True)
    
    sentiment_data = []
    
    for ticker in TICKERS:
        ticker_prices = prices_df[prices_df['Ticker'] == ticker].copy()
        
        for date in ticker_prices.index:
            # Simuler du sentiment basé sur les mouvements de prix (pour la démo)
            # Dans la réalité, le sentiment viendrait de sources textuelles
            
            # Nombre d'articles/tweets par jour (aléatoire)
            num_mentions = np.random.poisson(15) + 5
            
            # Sentiment de base avec un peu de bruit
            base_sentiment = np.random.normal(0, 0.3)
            
            # Ajouter une corrélation légère avec les retours passés
            if len(ticker_prices.loc[:date]) > 1:
                recent_return = ticker_prices.loc[:date, 'Close'].pct_change().iloc[-1]
                # Le sentiment tend à suivre les mouvements de prix
                sentiment_bias = recent_return * 2 if not pd.isna(recent_return) else 0
                base_sentiment += sentiment_bias
            
            # Clip entre -1 et 1
            base_sentiment = np.clip(base_sentiment, -1, 1)
            
            # Générer des sentiments individuels autour de la moyenne
            individual_sentiments = np.random.normal(base_sentiment, 0.2, num_mentions)
            individual_sentiments = np.clip(individual_sentiments, -1, 1)
            
            # Agréger
            sentiment_data.append({
                'Date': date,
                'Ticker': ticker,
                'Sentiment_Mean': individual_sentiments.mean(),
                'Sentiment_Std': individual_sentiments.std(),
                'Sentiment_Min': individual_sentiments.min(),
                'Sentiment_Max': individual_sentiments.max(),
                'Num_Mentions': num_mentions,
                'Positive_Count': (individual_sentiments > 0.1).sum(),
                'Negative_Count': (individual_sentiments < -0.1).sum(),
                'Neutral_Count': ((individual_sentiments >= -0.1) & (individual_sentiments <= 0.1)).sum()
            })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    
    # Sauvegarder
    output_file = raw_dir / 'sentiment_data.csv'
    sentiment_df.to_csv(output_file, index=False)
    print(f"✓ Données de sentiment sauvegardées dans {output_file}")
    print(f"  Total: {len(sentiment_df)} lignes")
    
    return sentiment_df

def download_sample_news_dataset():
    """
    Instructions pour télécharger un vrai dataset de news financières
    """
    print("\n📚 Dataset de news financières recommandés:")
    print("\n  Option 1 - Kaggle:")
    print("  • 'Daily Financial News for 6000+ Stocks'")
    print("    https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
    print("\n  Option 2 - Kaggle:")
    print("  • 'Financial Sentiment Analysis'")
    print("    https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis")
    print("\n  Pour utiliser un vrai dataset:")
    print("  1. Télécharger depuis Kaggle")
    print("  2. Placer dans data/raw/")
    print("  3. Adapter le script de preprocessing")

def create_sample_combined_dataset():
    """Crée un fichier exemple combinant prix et sentiment"""
    print("\n🔗 Création d'un dataset combiné exemple...")
    
    raw_dir = DATA_DIR / 'raw'
    processed_dir = DATA_DIR / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Charger les données
    prices = pd.read_csv(raw_dir / 'stock_prices.csv', index_col=0, parse_dates=True)
    sentiment = pd.read_csv(raw_dir / 'sentiment_data.csv', parse_dates=['Date'])
    
    # Merge
    prices_reset = prices.reset_index().rename(columns={'index': 'Date'})
    combined = pd.merge(prices_reset, sentiment, on=['Date', 'Ticker'], how='left')
    
    # Sauvegarder
    output_file = processed_dir / 'combined_data.csv'
    combined.to_csv(output_file, index=False)
    print(f"✓ Dataset combiné sauvegardé dans {output_file}")
    
    return combined

def main():
    """Fonction principale"""
    print("="*60)
    print("  Téléchargement des données - Stock Sentiment Prediction")
    print("="*60)
    
    # Créer les répertoires
    (DATA_DIR / 'raw').mkdir(exist_ok=True)
    (DATA_DIR / 'processed').mkdir(exist_ok=True)
    
    # Télécharger les prix
    prices_df = download_stock_data()
    
    # Générer du sentiment synthétique
    sentiment_df = generate_synthetic_sentiment()
    
    # Créer un dataset combiné
    combined_df = create_sample_combined_dataset()
    
    # Afficher des infos
    print("\n" + "="*60)
    print("📊 Résumé des données téléchargées")
    print("="*60)
    print(f"\nTickers: {', '.join(TICKERS)}")
    print(f"Période: {START_DATE} à {END_DATE}")
    print(f"Prix: {len(prices_df)} lignes")
    print(f"Sentiment: {len(sentiment_df)} lignes")
    print(f"Combiné: {len(combined_df)} lignes")
    
    # Instructions pour améliorer
    download_sample_news_dataset()
    
    print("\n✅ Téléchargement terminé!")
    print("\n💡 Prochaine étape: Ouvrir notebooks/01_data_collection.ipynb")

if __name__ == "__main__":
    main()
