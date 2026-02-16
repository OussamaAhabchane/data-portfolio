"""
Text Preprocessing pour analyse de reviews e-commerce
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Tuple

# NLP imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import spacy
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK or spaCy not available. Install with: pip install nltk spacy")


class TextPreprocessor:
    """
    Classe pour préprocesser des textes de reviews
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Args:
            use_spacy: Utiliser spaCy pour lemmatization (plus précis que NLTK)
        """
        self.use_spacy = use_spacy
        
        if NLTK_AVAILABLE:
            # Télécharger les ressources NLTK si nécessaire
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                print("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            
            self.lemmatizer = WordNetLemmatizer()
        
        if use_spacy and NLTK_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            except OSError:
                print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                print("Falling back to NLTK")
                self.use_spacy = False
        
        # Stopwords supplémentaires spécifiques e-commerce
        self.ecommerce_stopwords = {
            'product', 'item', 'bought', 'purchase', 'ordered', 'order',
            'amazon', 'buy', 'seller', 'shipping', 'delivery'
        }
        self.stop_words.update(self.ecommerce_stopwords)
    
    def clean_text(self, text: str) -> str:
        """
        Nettoyage basique du texte
        
        Args:
            text: Texte brut
            
        Returns:
            Texte nettoyé
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Supprimer les emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Gérer les contractions courantes
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Supprimer les chiffres (optionnel, peut être important pour sizing)
        # text = re.sub(r'\d+', '', text)
        
        # Supprimer la ponctuation excessive mais garder certains (!, ?)
        text = re.sub(r'[^\w\s!?]', ' ', text)
        
        # Supprimer les espaces multiples
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize le texte
        
        Args:
            text: Texte nettoyé
            
        Returns:
            Liste de tokens
        """
        if self.use_spacy:
            doc = self.nlp(text)
            return [token.text for token in doc]
        else:
            return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Supprime les stopwords
        
        Args:
            tokens: Liste de tokens
            
        Returns:
            Tokens filtrés
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize les tokens
        
        Args:
            tokens: Liste de tokens
            
        Returns:
            Tokens lemmatizés
        """
        if self.use_spacy:
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, 
                   remove_stops: bool = True,
                   lemmatize: bool = True) -> str:
        """
        Pipeline complet de preprocessing
        
        Args:
            text: Texte brut
            remove_stops: Supprimer les stopwords
            lemmatize: Appliquer la lemmatization
            
        Returns:
            Texte preprocessé (string)
        """
        # Nettoyer
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Filtrer les tokens trop courts
        tokens = [t for t in tokens if len(t) > 2]
        
        # Supprimer stopwords
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Rejoindre
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str], 
                        show_progress: bool = True) -> List[str]:
        """
        Preprocess un batch de textes
        
        Args:
            texts: Liste de textes
            show_progress: Afficher la progression
            
        Returns:
            Liste de textes preprocessés
        """
        if show_progress:
            try:
                from tqdm import tqdm
                texts = tqdm(texts, desc="Preprocessing texts")
            except ImportError:
                pass
        
        return [self.preprocess(text) for text in texts]


def extract_aspects(text: str, spacy_model=None) -> dict:
    """
    Extrait les aspects mentionnés dans une review (fit, quality, etc.)
    
    Args:
        text: Texte de la review
        spacy_model: Modèle spaCy chargé
        
    Returns:
        Dict des aspects détectés avec sentiments
    """
    if spacy_model is None:
        try:
            spacy_model = spacy.load('en_core_web_sm')
        except:
            return {}
    
    doc = spacy_model(text.lower())
    
    # Définir les patterns d'aspects
    aspect_keywords = {
        'fit': ['fit', 'fits', 'size', 'sizing', 'small', 'large', 'tight', 'loose'],
        'quality': ['quality', 'material', 'fabric', 'cheap', 'durable', 'sturdy'],
        'comfort': ['comfortable', 'comfort', 'soft', 'itchy', 'scratchy', 'cozy'],
        'style': ['style', 'stylish', 'beautiful', 'cute', 'ugly', 'fashionable'],
        'color': ['color', 'colour', 'shade', 'bright', 'dark', 'faded'],
        'delivery': ['delivery', 'shipping', 'arrived', 'package', 'fast', 'slow']
    }
    
    # Sentiment words (simple)
    positive_words = {'good', 'great', 'excellent', 'perfect', 'love', 'beautiful', 
                     'comfortable', 'soft', 'true', 'nice', 'happy'}
    negative_words = {'bad', 'poor', 'terrible', 'awful', 'hate', 'ugly', 
                     'uncomfortable', 'cheap', 'small', 'large', 'wrong'}
    
    aspects_found = {}
    
    for aspect, keywords in aspect_keywords.items():
        aspect_mentioned = False
        sentiment_score = 0
        
        for token in doc:
            if token.text in keywords:
                aspect_mentioned = True
                
                # Analyser le sentiment autour
                window = 3  # Fenêtre de mots
                start = max(0, token.i - window)
                end = min(len(doc), token.i + window + 1)
                
                context = doc[start:end]
                for ctx_token in context:
                    if ctx_token.text in positive_words:
                        sentiment_score += 1
                    elif ctx_token.text in negative_words:
                        sentiment_score -= 1
        
        if aspect_mentioned:
            aspects_found[aspect] = {
                'mentioned': True,
                'sentiment': 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
            }
    
    return aspects_found


def create_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    Crée des n-grams à partir de tokens
    
    Args:
        tokens: Liste de tokens
        n: Taille des n-grams (2 = bigrams, 3 = trigrams)
        
    Returns:
        Liste de n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = '_'.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams


# Exemple d'utilisation
if __name__ == "__main__":
    # Test
    preprocessor = TextPreprocessor(use_spacy=True)
    
    test_reviews = [
        "This dress is absolutely beautiful! The fit is perfect and the material quality is excellent.",
        "Terrible product. The size is way too small and the fabric feels cheap. Very disappointed!",
        "It's okay. Nothing special but nothing wrong either. Average quality for the price."
    ]
    
    print("Testing text preprocessor...\n")
    for i, review in enumerate(test_reviews, 1):
        print(f"Review {i}:")
        print(f"Original: {review}")
        preprocessed = preprocessor.preprocess(review)
        print(f"Preprocessed: {preprocessed}")
        print()
