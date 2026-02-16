"""
Feature Engineering pour prédiction boursière avec sentiment
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class FinancialFeatureEngineer:
    """
    Classe pour créer des features techniques et de prix
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame avec colonnes OHLCV
        """
        self.df = df.copy()
        self.df = self.df.sort_values('Date')
    
    def add_returns(self, periods: List[int] = [1, 3, 5, 7, 14]) -> pd.DataFrame:
        """
        Ajoute les returns (rendements) sur différentes périodes
        
        Args:
            periods: Périodes pour calculer les returns
            
        Returns:
            DataFrame avec colonnes de returns
        """
        for period in periods:
            col_name = f'Return_{period}d'
            self.df[col_name] = self.df.groupby('Ticker')['Close'].pct_change(period)
        
        return self.df
    
    def add_moving_averages(self, windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Ajoute les moyennes mobiles simples (SMA)
        
        Args:
            windows: Fenêtres pour les moyennes mobiles
            
        Returns:
            DataFrame avec colonnes SMA
        """
        for window in windows:
            col_name = f'SMA_{window}'
            self.df[col_name] = self.df.groupby('Ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Distance au SMA (en %)
            self.df[f'Dist_to_SMA_{window}'] = (
                (self.df['Close'] - self.df[col_name]) / self.df[col_name] * 100
            )
        
        return self.df
    
    def add_exponential_moving_averages(self, windows: List[int] = [12, 26]) -> pd.DataFrame:
        """
        Ajoute les moyennes mobiles exponentielles (EMA)
        
        Args:
            windows: Fenêtres pour les EMA
            
        Returns:
            DataFrame avec colonnes EMA
        """
        for window in windows:
            col_name = f'EMA_{window}'
            self.df[col_name] = self.df.groupby('Ticker')['Close'].transform(
                lambda x: x.ewm(span=window, adjust=False).mean()
            )
        
        return self.df
    
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Ajoute l'indicateur RSI (Relative Strength Index)
        
        Args:
            period: Période pour le calcul du RSI
            
        Returns:
            DataFrame avec colonne RSI
        """
        def calculate_rsi(prices, period):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        self.df['RSI'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: calculate_rsi(x, period)
        )
        
        return self.df
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Ajoute l'indicateur MACD
        
        Args:
            fast: Période rapide
            slow: Période lente
            signal: Période de signal
            
        Returns:
            DataFrame avec colonnes MACD
        """
        self.df[f'EMA_{fast}'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.ewm(span=fast, adjust=False).mean()
        )
        self.df[f'EMA_{slow}'] = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.ewm(span=slow, adjust=False).mean()
        )
        
        self.df['MACD'] = self.df[f'EMA_{fast}'] - self.df[f'EMA_{slow}']
        self.df['MACD_Signal'] = self.df.groupby('Ticker')['MACD'].transform(
            lambda x: x.ewm(span=signal, adjust=False).mean()
        )
        self.df['MACD_Diff'] = self.df['MACD'] - self.df['MACD_Signal']
        
        return self.df
    
    def add_bollinger_bands(self, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Ajoute les Bandes de Bollinger
        
        Args:
            window: Fenêtre pour le calcul
            num_std: Nombre d'écarts-types
            
        Returns:
            DataFrame avec colonnes Bollinger
        """
        rolling_mean = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        rolling_std = self.df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        
        self.df['BB_Middle'] = rolling_mean
        self.df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        self.df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
        
        # Position relative dans les bandes (0 = lower, 1 = upper)
        self.df['BB_Position'] = (
            (self.df['Close'] - self.df['BB_Lower']) / 
            (self.df['BB_Upper'] - self.df['BB_Lower'])
        )
        
        # Bandwidth
        self.df['BB_Width'] = (
            (self.df['BB_Upper'] - self.df['BB_Lower']) / self.df['BB_Middle']
        )
        
        return self.df
    
    def add_volatility(self, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Ajoute des mesures de volatilité
        
        Args:
            windows: Fenêtres pour calculer la volatilité
            
        Returns:
            DataFrame avec colonnes de volatilité
        """
        # Calcul des returns si pas déjà fait
        if 'Return_1d' not in self.df.columns:
            self.df['Return_1d'] = self.df.groupby('Ticker')['Close'].pct_change()
        
        for window in windows:
            # Volatilité historique (écart-type des returns)
            self.df[f'Volatility_{window}d'] = self.df.groupby('Ticker')['Return_1d'].transform(
                lambda x: x.rolling(window, min_periods=1).std() * np.sqrt(252)  # Annualisée
            )
        
        # True Range (pour ATR)
        self.df['TR'] = np.maximum(
            self.df['High'] - self.df['Low'],
            np.maximum(
                abs(self.df['High'] - self.df['Close'].shift(1)),
                abs(self.df['Low'] - self.df['Close'].shift(1))
            )
        )
        
        # Average True Range
        self.df['ATR_14'] = self.df.groupby('Ticker')['TR'].transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )
        
        return self.df
    
    def add_volume_features(self, windows: List[int] = [5, 20]) -> pd.DataFrame:
        """
        Ajoute des features basées sur le volume
        
        Args:
            windows: Fenêtres pour moyennes de volume
            
        Returns:
            DataFrame avec colonnes de volume
        """
        for window in windows:
            # Volume moyen
            self.df[f'Volume_MA_{window}'] = self.df.groupby('Ticker')['Volume'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Ratio volume actuel / volume moyen
            self.df[f'Volume_Ratio_{window}'] = (
                self.df['Volume'] / self.df[f'Volume_MA_{window}']
            )
        
        # Volume * Price (approximation de l'intérêt)
        self.df['Volume_Price'] = self.df['Volume'] * self.df['Close']
        
        return self.df
    
    def add_price_features(self) -> pd.DataFrame:
        """
        Ajoute des features basiques de prix
        
        Returns:
            DataFrame avec colonnes de prix
        """
        # High-Low spread
        self.df['HL_Spread'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        
        # Open-Close spread
        self.df['OC_Spread'] = (self.df['Close'] - self.df['Open']) / self.df['Open']
        
        # Écart au prix le plus haut récent (52 semaines)
        self.df['High_52w'] = self.df.groupby('Ticker')['High'].transform(
            lambda x: x.rolling(252, min_periods=1).max()
        )
        self.df['Dist_from_High_52w'] = (
            (self.df['Close'] - self.df['High_52w']) / self.df['High_52w'] * 100
        )
        
        # Écart au prix le plus bas récent
        self.df['Low_52w'] = self.df.groupby('Ticker')['Low'].transform(
            lambda x: x.rolling(252, min_periods=1).min()
        )
        self.df['Dist_from_Low_52w'] = (
            (self.df['Close'] - self.df['Low_52w']) / self.df['Low_52w'] * 100
        )
        
        return self.df
    
    def add_all_features(self) -> pd.DataFrame:
        """
        Ajoute toutes les features en une fois
        
        Returns:
            DataFrame complet avec toutes les features
        """
        print("Adding returns...")
        self.add_returns()
        
        print("Adding moving averages...")
        self.add_moving_averages()
        
        print("Adding exponential moving averages...")
        self.add_exponential_moving_averages()
        
        print("Adding RSI...")
        self.add_rsi()
        
        print("Adding MACD...")
        self.add_macd()
        
        print("Adding Bollinger Bands...")
        self.add_bollinger_bands()
        
        print("Adding volatility features...")
        self.add_volatility()
        
        print("Adding volume features...")
        self.add_volume_features()
        
        print("Adding price features...")
        self.add_price_features()
        
        print("✓ All features added!")
        
        return self.df


def create_target_variable(df: pd.DataFrame, 
                          target_type: str = 'direction',
                          horizon: int = 1) -> pd.DataFrame:
    """
    Crée la variable cible pour la prédiction
    
    Args:
        df: DataFrame avec prix
        target_type: 'direction' (up/down) ou 'return' (rendement)
        horizon: Nombre de jours dans le futur
        
    Returns:
        DataFrame avec colonne target
    """
    df = df.copy()
    
    if target_type == 'direction':
        # Prédire si le prix monte ou descend
        future_return = df.groupby('Ticker')['Close'].shift(-horizon) / df['Close'] - 1
        df['Target'] = (future_return > 0).astype(int)
        
    elif target_type == 'return':
        # Prédire le rendement exact
        df['Target'] = df.groupby('Ticker')['Close'].shift(-horizon) / df['Close'] - 1
    
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    return df


def merge_sentiment_and_price(price_df: pd.DataFrame, 
                             sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge les données de prix et de sentiment
    
    Args:
        price_df: DataFrame avec prix et features techniques
        sentiment_df: DataFrame avec sentiment agrégé
        
    Returns:
        DataFrame combiné
    """
    # S'assurer que les dates sont au bon format
    if 'Date' not in price_df.columns and price_df.index.name == 'Date':
        price_df = price_df.reset_index()
    
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.date
    
    # Merge
    merged = pd.merge(
        price_df,
        sentiment_df,
        on=['Date', 'Ticker'],
        how='left'
    )
    
    # Fill NaN pour les jours sans sentiment (weekends, etc.)
    sentiment_cols = [col for col in sentiment_df.columns if col not in ['Date', 'Ticker']]
    merged[sentiment_cols] = merged.groupby('Ticker')[sentiment_cols].fillna(method='ffill')
    
    return merged


# Test
if __name__ == "__main__":
    # Exemple d'utilisation
    print("Feature Engineering module loaded successfully!")
    print("\nUsage example:")
    print("  fe = FinancialFeatureEngineer(df)")
    print("  df_with_features = fe.add_all_features()")
