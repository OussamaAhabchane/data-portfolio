"""
Module d'analyse de sentiment pour données textuelles financières
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union
import re

# NLP imports
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    print("Warning: Advanced NLP libraries not available. Install transformers, vaderSentiment, textblob")


class SentimentAnalyzer:
    """
    Classe pour analyser le sentiment de textes financiers
    Supporte plusieurs méthodes: TextBlob, VADER, FinBERT
    """
    
    def __init__(self, method='vader'):
        """
        Args:
            method: 'textblob', 'vader', ou 'finbert'
        """
        self.method = method
        
        if method == 'vader' and ADVANCED_NLP_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        elif method == 'finbert' and ADVANCED_NLP_AVAILABLE:
            self._load_finbert()
    
    def _load_finbert(self):
        """Charge le modèle FinBERT"""
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            print("✓ FinBERT model loaded")
        except Exception as e:
            print(f"Warning: Could not load FinBERT: {e}")
            print("Falling back to VADER")
            self.method = 'vader'
            self.vader = SentimentIntensityAnalyzer()
    
    def preprocess_text(self, text: str) -> str:
        """
        Nettoyage basique du texte
        
        Args:
            text: Texte brut
            
        Returns:
            Texte nettoyé
        """
        if not isinstance(text, str):
            return ""
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Supprimer les mentions @ et hashtags (garder le texte)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        
        # Supprimer les caractères spéciaux excessifs
        text = re.sub(r'[^\w\s.,!?$%]', '', text)
        
        # Normaliser les espaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyse avec TextBlob
        
        Returns:
            Dict avec 'polarity' [-1, 1] et 'subjectivity' [0, 1]
        """
        blob = TextBlob(text)
        return {
            'sentiment': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyse avec VADER (optimisé pour textes courts)
        
        Returns:
            Dict avec scores de sentiment
        """
        scores = self.vader.polarity_scores(text)
        return {
            'sentiment': scores['compound'],  # Score principal [-1, 1]
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyse avec FinBERT (modèle spécialisé finance)
        
        Returns:
            Dict avec sentiment et probabilités
        """
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT classes: [positive, negative, neutral]
        labels = ['positive', 'negative', 'neutral']
        probs_dict = {labels[i]: probs[0][i].item() for i in range(3)}
        
        # Calculer un score composite [-1, 1]
        sentiment = probs_dict['positive'] - probs_dict['negative']
        
        return {
            'sentiment': sentiment,
            'positive_prob': probs_dict['positive'],
            'negative_prob': probs_dict['negative'],
            'neutral_prob': probs_dict['neutral']
        }
    
    def analyze(self, text: str, preprocess: bool = True) -> Dict[str, float]:
        """
        Analyse le sentiment d'un texte avec la méthode choisie
        
        Args:
            text: Texte à analyser
            preprocess: Appliquer le preprocessing
            
        Returns:
            Dict avec scores de sentiment
        """
        if preprocess:
            text = self.preprocess_text(text)
        
        if not text:
            return {'sentiment': 0.0}
        
        if self.method == 'textblob':
            return self.analyze_textblob(text)
        elif self.method == 'vader':
            return self.analyze_vader(text)
        elif self.method == 'finbert':
            return self.analyze_finbert(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def analyze_batch(self, texts: List[str], 
                     show_progress: bool = True) -> pd.DataFrame:
        """
        Analyse un batch de textes
        
        Args:
            texts: Liste de textes
            show_progress: Afficher la progression
            
        Returns:
            DataFrame avec les résultats
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Analyzing sentiment")
            except ImportError:
                pass
        
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        
        return pd.DataFrame(results)


def aggregate_daily_sentiment(df: pd.DataFrame, 
                              date_col: str = 'Date',
                              ticker_col: str = 'Ticker',
                              sentiment_col: str = 'sentiment') -> pd.DataFrame:
    """
    Agrège les sentiments à l'échelle journalière
    
    Args:
        df: DataFrame avec colonnes Date, Ticker, sentiment
        date_col: Nom de la colonne date
        ticker_col: Nom de la colonne ticker
        sentiment_col: Nom de la colonne sentiment
        
    Returns:
        DataFrame agrégé par jour et ticker
    """
    # Convertir la date si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Extraire juste la date (sans heure)
    df['Date_Only'] = df[date_col].dt.date
    
    # Agrégation
    agg_dict = {
        sentiment_col: ['mean', 'std', 'min', 'max', 'count'],
    }
    
    # Compter positif/négatif/neutre
    df['is_positive'] = (df[sentiment_col] > 0.1).astype(int)
    df['is_negative'] = (df[sentiment_col] < -0.1).astype(int)
    df['is_neutral'] = ((df[sentiment_col] >= -0.1) & (df[sentiment_col] <= 0.1)).astype(int)
    
    agg_dict.update({
        'is_positive': 'sum',
        'is_negative': 'sum',
        'is_neutral': 'sum'
    })
    
    grouped = df.groupby(['Date_Only', ticker_col]).agg(agg_dict).reset_index()
    
    # Flatten multi-level columns
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in grouped.columns.values]
    
    # Renommer pour clarté
    rename_dict = {
        f'{sentiment_col}_mean': 'Sentiment_Mean',
        f'{sentiment_col}_std': 'Sentiment_Std',
        f'{sentiment_col}_min': 'Sentiment_Min',
        f'{sentiment_col}_max': 'Sentiment_Max',
        f'{sentiment_col}_count': 'Num_Mentions',
        'is_positive_sum': 'Positive_Count',
        'is_negative_sum': 'Negative_Count',
        'is_neutral_sum': 'Neutral_Count',
        'Date_Only': 'Date'
    }
    
    grouped = grouped.rename(columns=rename_dict)
    
    return grouped


def create_sentiment_features(df: pd.DataFrame, 
                              windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
    """
    Crée des features supplémentaires à partir du sentiment
    
    Args:
        df: DataFrame avec sentiment quotidien
        windows: Fenêtres de rolling pour features
        
    Returns:
        DataFrame avec features additionnelles
    """
    df = df.copy()
    df = df.sort_values('Date')
    
    for window in windows:
        # Rolling mean
        df[f'Sentiment_MA{window}'] = df.groupby('Ticker')['Sentiment_Mean'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Rolling std (volatilité du sentiment)
        df[f'Sentiment_Vol{window}'] = df.groupby('Ticker')['Sentiment_Mean'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        
        # Cumulative sum
        df[f'Sentiment_Cumsum{window}'] = df.groupby('Ticker')['Sentiment_Mean'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
    
    # Ratio positif/négatif
    df['Pos_Neg_Ratio'] = (df['Positive_Count'] + 1) / (df['Negative_Count'] + 1)
    
    # Changement de sentiment
    df['Sentiment_Change'] = df.groupby('Ticker')['Sentiment_Mean'].diff()
    
    return df


# Exemple d'utilisation
if __name__ == "__main__":
    # Test rapide
    analyzer = SentimentAnalyzer(method='vader')
    
    test_texts = [
        "Apple stock surges on strong earnings report!",
        "Tesla faces challenges amid increased competition",
        "Amazon announces major expansion plans",
    ]
    
    print("Testing sentiment analyzer...\n")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']:.3f}\n")
