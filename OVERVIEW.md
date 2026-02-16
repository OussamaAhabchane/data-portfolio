# 📊 Portfolio Data Science Python - Vue d'Ensemble

---

## 📁 Structure Complète

```
data-portfolio/
│
├── README.md                    ⭐ Page d'accueil du portfolio
├── QUICK_START.md              🚀 Guide de démarrage rapide
├── requirements.txt             📦 Dépendances globales
├── setup.py                     🔧 Script d'installation automatique
├── .gitignore                   🚫 Fichiers à ignorer dans git
│
├── 01-stock-sentiment-prediction/
│   ├── README.md               📖 Documentation complète du projet
│   ├── requirements.txt         📦 Dépendances spécifiques
│   ├── data/
│   │   ├── raw/                💾 Données brutes
│   │   ├── processed/          ✨ Données préprocessées
│   │   └── download_data.py    ⬇️  Script de téléchargement
│   ├── notebooks/              📓 Notebooks Jupyter (à créer)
│   ├── src/
│   │   ├── sentiment_analyzer.py    🎭 Analyse de sentiment
│   │   ├── feature_engineering.py   🔨 Feature engineering
│   │   └── (autres modules...)
│   └── visualizations/         📊 Graphiques générés
│
├── 02-fraud-detection/
│   ├── README.md               📖 Doc détection de fraude
│   ├── requirements.txt
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   ├── src/
│   │   └── train_model.py      🤖 Pipeline ML complet
│   └── models/                 💾 Modèles sauvegardés
│
├── 03-ecommerce-review-analysis/
│   ├── README.md               📖 Doc analyse NLP
│   ├── requirements.txt
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   │   └── preprocessing.py    📝 Preprocessing NLP avancé
│   └── models/
│
└── 04-customer-churn-prediction/
    ├── README.md               📖 Doc prédiction churn
    ├── requirements.txt
    ├── data/
    ├── notebooks/
    ├── src/
    │   └── rfm.py              📈 Segmentation RFM
    └── models/
```

---

## 🎯 Les 4 Projets en Détail

### 📈 Projet 1 : Stock Sentiment Prediction
**Niveau** : Intermédiaire | **Durée estimée** : 2-3 semaines

**Compétences démontrées** :
- ✅ Collecte de données financières (yfinance)
- ✅ Analyse de sentiment (TextBlob, VADER, FinBERT)
- ✅ Feature engineering avancé (indicateurs techniques)
- ✅ Time series forecasting
- ✅ Combinaison de données textuelles et numériques

**Fichiers clés** :
- `sentiment_analyzer.py` : 3 méthodes de sentiment (600+ lignes)
- `feature_engineering.py` : 20+ indicateurs techniques (500+ lignes)
- `download_data.py` : Téléchargement automatique

**Dataset** : ✅ Auto-généré (Yahoo Finance + sentiment synthétique)

---

### 🚨 Projet 2 : Fraud Detection
**Niveau** : Intermédiaire | **Durée estimée** : 1-2 semaines

**Compétences démontrées** :
- ✅ Gestion de données déséquilibrées (<1% fraudes)
- ✅ SMOTE, undersampling, oversampling
- ✅ XGBoost, Random Forest, Isolation Forest
- ✅ Métriques adaptées (Precision-Recall, ROC-AUC)
- ✅ Feature importance et SHAP values

**Fichiers clés** :
- `train_model.py` : Pipeline complet (400+ lignes)
- Classe `FraudDetectionPipeline` avec tout le workflow

**Dataset** : 📥 Kaggle - Credit Card Fraud Detection

---

### 🛍️ Projet 3 : E-commerce Review Analysis
**Niveau** : Intermédiaire-Avancé | **Durée estimée** : 2-3 semaines

**Compétences démontrées** :
- ✅ NLP preprocessing complet
- ✅ Sentiment analysis multi-classes
- ✅ Topic modeling (LDA)
- ✅ Système de recommandation
- ✅ Aspect-based sentiment analysis

**Fichiers clés** :
- `preprocessing.py` : TextPreprocessor class complète (400+ lignes)
- Fonctions d'extraction d'aspects

**Dataset** : 📥 Kaggle - Women's E-commerce Clothing Reviews

---

### 📉 Projet 4 : Customer Churn Prediction
**Niveau** : Intermédiaire | **Durée estimée** : 2 semaines

**Compétences démontrées** :
- ✅ Prédiction de churn
- ✅ Segmentation RFM complète
- ✅ Customer Lifetime Value (CLV)
- ✅ Feature engineering métier
- ✅ Recommandations business actionnables

**Fichiers clés** :
- `rfm.py` : Classe RFMAnalyzer complète (500+ lignes)
- 12 segments clients avec stratégies

**Dataset** : 📥 Kaggle - Telco Customer Churn

---

## 💻 Technologies Utilisées

### Core Data Science
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **scikit-learn** : Machine Learning

### Machine Learning Avancé
- **XGBoost** : Gradient boosting
- **LightGBM** : ML rapide
- **CatBoost** : Catégories natives
- **imbalanced-learn** : Données déséquilibrées

### NLP
- **NLTK** : Traitement de texte
- **spaCy** : NLP avancé
- **Gensim** : Topic modeling
- **Transformers** : BERT, FinBERT

### Visualisation
- **Matplotlib** : Graphiques
- **Seaborn** : Viz statistiques
- **Plotly** : Viz interactives

### Finance
- **yfinance** : Données boursières
- **ta** : Technical analysis

### Explainability
- **SHAP** : Interprétabilité
- **LIME** : Explications locales

---

## 📊 Statistiques du Portfolio

### Lignes de Code
- **Projet 1** : ~1,500 lignes
- **Projet 2** : ~600 lignes
- **Projet 3** : ~700 lignes
- **Projet 4** : ~800 lignes
- **Total** : ~3,600+ lignes de code Python

### Documentation
- **READMEs** : 5 fichiers (main + 4 projets)
- **Guides** : QUICK_START.md
- **Total documentation** : ~5,000 mots

### Fichiers
- **Scripts Python** : 8 modules principaux
- **Requirements** : 5 fichiers (global + par projet)
- **Notebooks à créer** : ~15-20 notebooks

---

## 🎓 Compétences CV

Ce portfolio démontre :

### Hard Skills
✅ Python (pandas, numpy, scikit-learn)
✅ Machine Learning (Classification, Regression, Clustering)
✅ Deep Learning (Transformers, BERT)
✅ NLP (Sentiment Analysis, Topic Modeling)
✅ Time Series Forecasting
✅ Feature Engineering
✅ Data Visualization
✅ Model Evaluation & Selection
✅ Handling Imbalanced Data
✅ SQL & Data Manipulation

### Soft Skills
✅ Problem Solving
✅ Documentation
✅ Code Organization
✅ Business Acumen
✅ Communication (READMEs détaillés)

### Domaines d'application
✅ Finance (trading, fraud)
✅ Marketing (churn, CLV)
✅ E-commerce (reviews, recommandation)
✅ Text Mining

---

## 🚀 Comment Utiliser Ce Portfolio

### 1. Installation Rapide
```bash
cd data-portfolio
python setup.py
```

### 2. Choisir un Projet
```bash
cd 01-stock-sentiment-prediction
```

### 3. Télécharger les Données
```bash
python data/download_data.py
```

### 4. Lancer Jupyter
```bash
jupyter notebook notebooks/
```

### 5. Pusher sur GitHub
```bash
git init
git add .
git commit -m "Data Science Portfolio"
git push
```

---

## 📈 Progression Suggérée

### Semaine 1-2 : Projet 1 (Stock Sentiment)
- Setup et exploration
- Feature engineering
- Modélisation

### Semaine 3 : Projet 2 (Fraud Detection)
- Dataset Kaggle
- SMOTE et sampling
- Comparaison modèles

### Semaine 4-5 : Projet 3 (E-commerce NLP)
- Preprocessing NLP
- Topic modeling
- Recommandations

### Semaine 6 : Projet 4 (Customer Churn)
- Segmentation RFM
- Prédiction churn
- CLV analysis

### Semaine 7-8 : Finition
- Visualisations
- Documentation
- Publication GitHub

---

## 🎯 Objectifs Atteints

✅ **4 projets complets** couvrant Finance et NLP
✅ **Code professionnel** bien structuré et commenté
✅ **Documentation exhaustive** avec READMEs détaillés
✅ **Scripts réutilisables** et modulaires
✅ **Best practices** (requirements, .gitignore, structure)
✅ **Diversité technique** (ML, DL, NLP, Time Series)
✅ **Datasets réels** de Kaggle et APIs

---

## 🌟 Points Forts

1. **Niveau intermédiaire** : Parfait pour candidatures mid-level
2. **Documenté** : Chaque projet a son README complet
3. **Reproductible** : Scripts de téléchargement et setup
4. **Professionnel** : Structure claire, code propre
5. **Diversifié** : Finance, Marketing, NLP
6. **Actionnable** : Insights business dans chaque projet
