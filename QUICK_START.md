# 🚀 Guide de Démarrage Rapide - Portfolio Data Science

## 📦 Contenu du Portfolio

**4 projets complets** prêts à être utilisés :

### 1️⃣ Stock Sentiment Prediction (Finance + NLP)
- **Thématique** : Analyse de sentiment financier et prédiction de prix
- **Dataset** : Yahoo Finance + Sentiment synthétique
- **Compétences** : Time series, NLP, Feature engineering
- **Fichiers clés** : 
  - `data/download_data.py` - Télécharge les données boursières
  - `src/sentiment_analyzer.py` - Analyse de sentiment
  - `src/feature_engineering.py` - Features techniques avancées

### 2️⃣ Fraud Detection (Finance + ML)
- **Thématique** : Détection de fraudes bancaires
- **Dataset** : Kaggle Credit Card Fraud
- **Compétences** : Données déséquilibrées, SMOTE, XGBoost
- **Fichiers clés** :
  - `src/train_model.py` - Pipeline complet de ML
  - Comparaison de multiples approches de sampling

### 3️⃣ E-commerce Review Analysis (NLP)
- **Thématique** : Analyse de sentiment et Topic Modeling
- **Dataset** : Kaggle Women's E-commerce Reviews
- **Compétences** : NLP avancé, LDA, Recommendation
- **Fichiers clés** :
  - `src/preprocessing.py` - Preprocessing NLP complet
  - Topic modeling et système de recommandation

### 4️⃣ Customer Churn Prediction (Finance + Marketing)
- **Thématique** : Prédiction d'attrition client et CLV
- **Dataset** : Kaggle Telco Churn
- **Compétences** : Classification, RFM, Feature engineering
- **Fichiers clés** :
  - `src/rfm.py` - Segmentation RFM complète
  - Analyse de valeur client

---

## 🎯 Comment Utiliser ce Portfolio

### Option 1 : Upload sur GitHub (Recommandé)

```bash
# 1. Créer un nouveau repo sur GitHub
# 2. Dans votre terminal local :
cd data-portfolio
git init
git add .
git commit -m "Initial commit - Data Science Portfolio"
git branch -M main
git remote add origin https://github.com/VOTRE-USERNAME/data-portfolio.git
git push -u origin main
```

### Option 2 : Travailler localement

```bash
# 1. Extraire le dossier data-portfolio
# 2. Ouvrir un terminal dans le dossier
cd data-portfolio

# 3. Exécuter le setup automatique
python setup.py

# 4. Activer l'environnement virtuel
# Sur Mac/Linux :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# 5. Choisir un projet
cd 01-stock-sentiment-prediction

# 6. Télécharger les données
python data/download_data.py

# 7. Lancer Jupyter
jupyter notebook notebooks/
```
