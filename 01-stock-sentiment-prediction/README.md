# 📈 Stock Sentiment Prediction

Prédiction des mouvements boursiers en combinant l'analyse de sentiment de données textuelles financières (news, tweets) avec des indicateurs techniques.

## 🎯 Objectifs

1. Collecter et analyser le sentiment de données textuelles financières
2. Combiner sentiment et indicateurs techniques pour créer des features prédictives
3. Construire un modèle de prédiction des mouvements de prix
4. Évaluer l'impact du sentiment sur la performance boursière

## 📊 Dataset

**Sources** :
- **Yahoo Finance** : Données historiques de prix (AAPL, TSLA, AMZN, GOOGL)
- **FinancialNewsAPI** / **Twitter** : Articles de news et tweets financiers
- **Alternative** : Dataset Kaggle "Stock News Sentiment"

**Période** : 2020-2024

**Variables** :
- Prix : Open, High, Low, Close, Volume, Adjusted Close
- Indicateurs techniques : SMA, EMA, RSI, MACD, Bollinger Bands
- Sentiment : Score de sentiment [-1, 1], subjectivité, volume de mentions

## 🛠️ Technologies utilisées

```
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
yfinance >= 0.2.0
scikit-learn >= 1.0.0
nltk >= 3.6
transformers >= 4.20.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
plotly >= 5.0.0
textblob >= 0.17.0
```

## 📁 Structure du projet

```
01-stock-sentiment-prediction/
├── data/
│   ├── raw/                 # Données brutes téléchargées
│   ├── processed/           # Données nettoyées et features
│   └── download_data.py     # Script de téléchargement
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_collection.py   # Fonctions de collecte
│   ├── sentiment_analyzer.py # Analyse de sentiment
│   ├── feature_engineering.py # Création de features
│   ├── models.py            # Modèles ML
│   └── visualization.py     # Fonctions de visualisation
├── visualizations/          # Graphiques générés
├── requirements.txt
└── README.md
```

## 🚀 Installation et utilisation

### 1. Installation

```bash
cd 01-stock-sentiment-prediction
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon punkt stopwords
```

### 2. Téléchargement des données

```bash
python data/download_data.py
```

### 3. Exécution de l'analyse

Option A - Notebooks interactifs :
```bash
jupyter notebook notebooks/
```

Option B - Scripts :
```bash
python src/main.py --ticker AAPL --start 2020-01-01 --end 2024-01-01
```

## 🔍 Méthodologie

### 1. Collecte de données
- Téléchargement des prix historiques via yfinance
- Collecte de news/tweets via API ou dataset Kaggle
- Période d'entraînement : 2020-2023 / Test : 2024

### 2. Analyse de sentiment
- **TextBlob** : Baseline rapide
- **VADER** : Optimisé pour textes courts (tweets)
- **FinBERT** : Modèle BERT fine-tuné sur textes financiers
- Agrégation quotidienne : moyenne, écart-type, volume

### 3. Feature Engineering

**Indicateurs techniques** :
- SMA (20, 50, 200 jours)
- EMA (12, 26 jours)
- RSI (14 jours)
- MACD
- Bollinger Bands
- Volume moving average

**Features de sentiment** :
- Sentiment score moyen quotidien
- Écart-type du sentiment
- Nombre de mentions/articles
- Sentiment cumulé sur 3, 7 jours
- Ratio sentiment positif/négatif

**Features de prix** :
- Returns (1, 3, 7 jours)
- Volatilité
- High-Low spread
- Volume relatif

### 4. Modélisation

**Target** : Direction du prix (up/down) à J+1 ou rendement à J+1

**Modèles comparés** :
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LSTM (séquences temporelles)

**Validation** : Time Series Split (pas de shuffle)

**Métriques** :
- Accuracy
- Precision/Recall
- ROC-AUC
- Profit simulé (stratégie de trading)

## 📈 Résultats attendus

- **Baseline (indicateurs techniques seuls)** : ~55% accuracy
- **Avec sentiment** : ~60-65% accuracy
- **Analyse de corrélation** : Sentiment vs Returns
- **Feature importance** : Top features prédictives

## 📊 Visualisations clés

1. **Série temporelle** : Prix + Sentiment overlay
2. **Corrélation heatmap** : Features vs Returns
3. **Distribution de sentiment** : Par période et par action
4. **ROC Curves** : Comparaison des modèles
5. **Feature importance** : Top 20 features
6. **Confusion matrices**
7. **Cumulative returns** : Stratégie ML vs Buy & Hold

## 🎓 Apprentissages clés

- Combinaison de données textuelles et numériques
- Gestion de séries temporelles (pas de data leakage)
- NLP appliqué à la finance
- Feature engineering créatif
- Évaluation réaliste avec time series split

## ⚠️ Limitations

- Pas de données intraday (uniquement daily)
- Sentiment basé sur sources publiques (pas de données propriétaires)
- Pas de prise en compte des frais de transaction
- Performance passée ≠ performance future

## 🔮 Améliorations futures

1. Ajouter des données alternatives (Google Trends, Reddit WSB)
2. Inclure des features macro-économiques
3. Tester des architectures de deep learning avancées (Attention, Transformers)
4. Backtesting complet avec stratégie de trading
5. Dashboard interactif Streamlit

## 📚 Références

- FinBERT: https://github.com/ProsusAI/finBERT
- VADER Sentiment: https://github.com/cjhutto/vaderSentiment
- Technical Indicators: https://github.com/bukosabino/ta

## 📝 Licence

MIT License
