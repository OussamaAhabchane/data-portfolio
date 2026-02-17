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

---

## 📋 Installation Manuelle (si setup.py ne fonctionne pas)

### Étape 1 : Créer l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OU
venv\Scripts\activate  # Windows
```

### Étape 2 : Installer les dépendances
```bash
# Installation globale
pip install -r requirements.txt

# OU installation par projet
cd 01-stock-sentiment-prediction
pip install -r requirements.txt
```

### Étape 3 : Télécharger ressources NLP (pour projet 3)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

---

## 📊 Téléchargement des Datasets

### Projet 1 - Stock Sentiment
✅ **Automatique** : Le script télécharge via yfinance
```bash
python data/download_data.py
```

### Projet 2 - Fraud Detection
📥 **Manuel** : Télécharger depuis Kaggle
1. Aller sur : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Télécharger `creditcard.csv`
3. Placer dans `02-fraud-detection/data/raw/`

### Projet 3 - E-commerce Reviews
📥 **Manuel** : Télécharger depuis Kaggle
1. Aller sur : https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
2. Télécharger le CSV
3. Placer dans `03-ecommerce-review-analysis/data/raw/`

### Projet 4 - Customer Churn
📥 **Manuel** : Télécharger depuis Kaggle
1. Aller sur : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Télécharger `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Placer dans `04-customer-churn-prediction/data/raw/`

---

## 🎨 Personnalisation pour Votre Profil

### 1. Mettre à jour le README principal
Éditez `README.md` et remplacez :
- `[votre-username]` → Votre nom d'utilisateur GitHub
- `[votre-profil]` → Lien vers votre LinkedIn
- `votre.email@example.com` → Votre email

### 2. Ajouter vos résultats
Après avoir exécuté les projets :
- Remplacez les résultats "attendus" par vos résultats réels
- Ajoutez vos propres visualisations
- Documentez vos insights

### 3. Personnaliser les analyses
- Testez différents hyperparamètres
- Ajoutez vos propres features
- Créez des visualisations supplémentaires

---

## 💡 Conseils pour les Recruteurs

### Structurez votre présentation :

1. **README accrocheur** ✅ (déjà fait)
2. **Notebooks bien commentés** 📝 (à faire dans Jupyter)
3. **Code propre et modulaire** ✅ (déjà structuré)
4. **Visualisations professionnelles** 📊 (à générer)
5. **Documentation complète** 📚 (déjà fournie)

### Mettez en avant :
- ✨ **Compétences techniques** : Listées dans chaque README
- 📈 **Résultats quantifiables** : Métriques et performances
- 💼 **Business impact** : Insights actionnables
- 🔧 **Best practices** : Code modulaire, tests, documentation

---

## 🐛 Troubleshooting

### Erreur : "Module not found"
```bash
pip install [nom-du-module]
# OU réinstaller tous les requirements
pip install -r requirements.txt
```

### Erreur : "NLTK resources not found"
```bash
python -c "import nltk; nltk.download('all')"
```

### Erreur : "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### Problème de mémoire (dataset trop gros)
```python
# Lire seulement une partie du dataset
df = pd.read_csv('data.csv', nrows=10000)
```

### Jupyter ne démarre pas
```bash
pip install --upgrade jupyter notebook
jupyter notebook
```

---

## 📚 Ressources Supplémentaires

### Apprentissage
- **Kaggle Learn** : https://www.kaggle.com/learn
- **Fast.ai** : https://www.fast.ai/
- **Coursera ML** : https://www.coursera.org/learn/machine-learning

### Datasets
- **Kaggle** : https://www.kaggle.com/datasets
- **UCI ML Repository** : https://archive.ics.uci.edu/ml/
- **Data.gov** : https://www.data.gov/

### Documentation
- **Scikit-learn** : https://scikit-learn.org/
- **Pandas** : https://pandas.pydata.org/
- **Matplotlib** : https://matplotlib.org/

---

## ✅ Checklist avant de publier sur GitHub

- [ ] Remplacer les placeholders dans README.md
- [ ] Générer et ajouter des visualisations
- [ ] Tester que les notebooks s'exécutent
- [ ] Vérifier que .gitignore fonctionne (pas de gros fichiers)
- [ ] Ajouter une LICENSE (MIT recommandée)
- [ ] Créer des badges pour le README (optional)
- [ ] Ajouter des screenshots des visualisations
- [ ] Documenter vos résultats finaux

---

## 🎯 Prochaines Étapes

### Semaine 1-2 : Setup et exploration
- ✅ Installation complète
- ✅ Téléchargement des datasets
- 📊 Exécution des notebooks d'exploration

### Semaine 3-4 : Modélisation
- 🤖 Entraînement des modèles
- 📈 Optimisation des hyperparamètres
- 📊 Génération des visualisations

### Semaine 5-6 : Documentation
- 📝 Documenter vos résultats
- 🎨 Créer des visualisations professionnelles
- 📚 Rédiger vos insights

### Semaine 7-8 : Publication
- 🐙 Push sur GitHub
- 💼 Ajouter à votre CV/LinkedIn
- 🎤 Préparer votre pitch

---

## 🆘 Besoin d'Aide ?

- **Documentation projet** : Voir README.md de chaque projet
- **Issues techniques** : Vérifier requirements et versions Python
- **Questions dataset** : Consulter la page Kaggle du dataset
- **Amélioration code** : Les scripts sont commentés et modulaires

---

## 🌟 Bonus : Améliorations Possibles

### Niveau Débutant
- Ajouter plus de visualisations
- Tester d'autres hyperparamètres
- Créer un rapport PDF automatique

### Niveau Intermédiaire
- Créer une API Flask/FastAPI
- Ajouter un dashboard Streamlit
- Implémenter du feature engineering avancé

### Niveau Avancé
- Deep Learning (LSTM, Transformers)
- MLOps (MLflow, DVC)
- Déploiement cloud (AWS, GCP, Azure)
- CI/CD avec GitHub Actions

---

**Bon courage avec votre portfolio ! 🚀**

Si vous avez des questions, consultez d'abord les README des projets - tout est documenté en détail.
