# 📊 Portfolio Data Science - Python

Un portfolio complet de projets data science axés sur la **Finance** et le **NLP**, démontrant des compétences en analyse de données, machine learning et visualisation.

## 🎯 Compétences démontrées

- **Machine Learning** : Classification, Régression, Clustering
- **NLP** : Analyse de sentiment, Topic Modeling, Text Classification
- **Visualisation** : Matplotlib, Seaborn, Plotly
- **Feature Engineering** : Techniques avancées pour données structurées et textuelles
- **Gestion de données déséquilibrées** : SMOTE, Class Weighting
- **Déploiement** : Scripts reproductibles, documentation complète

## 📁 Structure du Portfolio

```
data-portfolio/
├── 01-stock-sentiment-prediction/    # Finance + NLP
├── 02-fraud-detection/                # Finance + ML
├── 03-ecommerce-review-analysis/      # NLP + Recommandation
├── 04-customer-churn-prediction/      # Finance + Marketing
└── README.md
```

## 🚀 Projets

### 1. Analyse de Sentiment et Prédiction des Cours Boursiers
**Thématiques** : Finance, NLP, Time Series

Combine l'analyse de sentiment de tweets/news financières avec des indicateurs techniques pour prédire les mouvements de prix des actions.

**Technologies** : Python, Pandas, Scikit-learn, NLTK, Transformers, yfinance

**Highlights** :
- Scraping et analyse de sentiment de données textuelles financières
- Feature engineering combinant sentiment et indicateurs techniques
- Modèles de prédiction avec validation temporelle
- Visualisations interactives des corrélations sentiment/prix

---

### 2. Détection de Fraude dans les Transactions Bancaires
**Thématiques** : Finance, Machine Learning, Data Imbalance

Système de détection de fraudes utilisant des techniques avancées pour gérer les données fortement déséquilibrées.

**Technologies** : Python, Pandas, Scikit-learn, Imbalanced-learn, XGBoost

**Highlights** :
- Traitement de datasets avec <1% de fraudes
- Feature engineering pour transactions financières
- Comparaison de multiple algorithmes (Random Forest, XGBoost, Isolation Forest)
- Métriques adaptées : Precision-Recall, ROC-AUC, F1-Score

---

### 3. Système de Recommandation et Analyse de Reviews E-commerce
**Thématiques** : NLP, Recommandation, Text Mining

Analyse approfondie de reviews clients avec topic modeling et système de recommandation basé sur le contenu.

**Technologies** : Python, NLTK, Gensim, Scikit-learn, SpaCy

**Highlights** :
- Preprocessing avancé de texte (lemmatization, stopwords)
- Topic modeling avec LDA pour identifier les thèmes récurrents
- Classification de sentiment multi-classes
- Système de recommandation de produits basé sur similarité textuelle

---

### 4. Prédiction de Churn et Analyse de Valeur Client
**Thématiques** : Finance, Marketing Analytics, Customer Intelligence

Prédiction de l'attrition client avec analyse de la valeur vie client (CLV) et segmentation RFM.

**Technologies** : Python, Pandas, Scikit-learn, Matplotlib, Seaborn

**Highlights** :
- Segmentation RFM (Recency, Frequency, Monetary)
- Modèles de prédiction de churn avec feature importance
- Calcul de Customer Lifetime Value
- Recommandations business actionnables

---

## 🛠️ Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Installation globale

```bash
# Cloner le repository
git clone https://github.com/votre-username/data-portfolio.git
cd data-portfolio

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer toutes les dépendances
pip install -r requirements.txt
```

### Installation par projet

Chaque projet contient son propre `requirements.txt`. Pour travailler sur un projet spécifique :

```bash
cd 01-stock-sentiment-prediction
pip install -r requirements.txt
```

## 📊 Datasets

Tous les projets utilisent des datasets publics et réels :
- **Kaggle** : Credit Card Fraud, E-commerce Reviews
- **Yahoo Finance** : Données boursières historiques
- **UCI Repository** : Customer Churn datasets
- **Twitter API** / **News APIs** : Données de sentiment

Les scripts de téléchargement/génération sont inclus dans chaque projet.

## 🎓 Compétences techniques

### Languages & Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Visualisation
- Matplotlib
- Seaborn
- Plotly
- Wordcloud

### Machine Learning
- Classification (Logistic Regression, Random Forest, XGBoost)
- Clustering (K-Means, DBSCAN)
- Feature Engineering
- Hyperparameter Tuning
- Cross-validation

### NLP
- NLTK
- SpaCy
- Gensim
- Transformers (BERT, DistilBERT)
- Sentiment Analysis
- Topic Modeling (LDA)

## 📈 Résultats clés

- **Fraude Detection** : 95%+ Recall sur détection de fraudes
- **Sentiment Analysis** : 82% accuracy sur prédiction de sentiment financier
- **Churn Prediction** : 88% F1-Score sur prédiction d'attrition
- **Topic Modeling** : Identification de 10 thèmes majeurs dans 50K+ reviews

## 📝 Licence

MIT License - Libre d'utilisation pour l'apprentissage et portfolio professionnel

## 👤 Contact

- **GitHub** : OussamaAhabchane
- **LinkedIn** : https://www.linkedin.com/in/oussama-ahabchane/
- **Email** : oussama.ahabchane@outlook.com

---

⭐ **Si ce portfolio vous a été utile, n'hésitez pas à le star !**
