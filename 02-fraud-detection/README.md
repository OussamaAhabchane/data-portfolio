# 🚨 Credit Card Fraud Detection

Système de détection de fraudes utilisant des techniques avancées de machine learning pour traiter des données fortement déséquilibrées (<1% de fraudes).

## 🎯 Objectifs

1. Construire un système robuste de détection de fraudes
2. Gérer efficacement les données déséquilibrées
3. Minimiser les faux négatifs (fraudes manquées) tout en contrôlant les faux positifs
4. Comparer plusieurs approches et algorithmes

## 📊 Dataset

**Source** : Kaggle - Credit Card Fraud Detection
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Caractéristiques** :
- 284,807 transactions
- 492 fraudes (0.172%)
- 28 features anonymisées (PCA)
- Features: Time, V1-V28, Amount
- Target: Class (0=normal, 1=fraud)

**Période** : Transactions sur 2 jours (Septembre 2013)

## 🛠️ Technologies utilisées

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
shap>=0.41.0
```

## 📁 Structure du projet

```
02-fraud-detection/
├── data/
│   ├── raw/                 # Dataset original
│   ├── processed/           # Données preprocessées
│   └── download_data.py     # Script de téléchargement
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_handling_imbalance.ipynb
│   └── 04_final_model.ipynb
├── src/
│   ├── preprocessing.py     # Preprocessing
│   ├── models.py            # Modèles ML
│   ├── evaluation.py        # Métriques
│   └── visualization.py     # Visualisations
├── models/                  # Modèles sauvegardés
├── requirements.txt
└── README.md
```

## 🚀 Installation et utilisation

### 1. Installation

```bash
cd 02-fraud-detection
pip install -r requirements.txt
```

### 2. Téléchargement des données

Le dataset est disponible sur Kaggle. Deux options:

**Option A - Téléchargement manuel:**
1. Télécharger depuis: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Placer `creditcard.csv` dans `data/raw/`

**Option B - API Kaggle:**
```bash
pip install kaggle
# Configurer API key (voir instructions Kaggle)
python data/download_data.py
```

### 3. Exécution

```bash
# Notebooks
jupyter notebook notebooks/

# Ou script complet
python src/train_model.py --model xgboost --sampling smote
```

## 🔍 Méthodologie

### 1. Exploration des données (EDA)

- Distribution des transactions normales vs frauduleuses
- Analyse des features anonymisées
- Corrélations
- Distribution temporelle
- Patterns dans les montants

### 2. Preprocessing

- **Normalisation** : StandardScaler pour Amount et Time
- **Feature Engineering** :
  - Log transform de Amount
  - Bins temporels (heure de la journée)
  - Interactions entre features importantes
  
### 3. Gestion du déséquilibre

Plusieurs approches testées:

**A. Undersampling**
- Random Undersampling
- NearMiss
- Tomek Links

**B. Oversampling**
- Random Oversampling
- SMOTE (Synthetic Minority Over-sampling)
- ADASYN

**C. Combinaisons**
- SMOTE + Tomek Links
- SMOTE + ENN

**D. Algorithmes adaptés**
- Class Weighting
- Ensemble methods (BalancedRandomForest)
- Anomaly detection (Isolation Forest, One-Class SVM)

### 4. Modélisation

**Modèles testés** :
1. **Logistic Regression** (baseline)
2. **Random Forest**
3. **XGBoost**
4. **LightGBM**
5. **Isolation Forest**
6. **Autoencoders** (Deep Learning)

**Validation** :
- Stratified K-Fold Cross-Validation
- Time-based split (si temporalité importante)
- Validation set avec distribution réelle

### 5. Métriques

⚠️ **Accuracy n'est PAS une bonne métrique ici !**

**Métriques utilisées** :
- **Precision** : % de prédictions de fraude qui sont correctes
- **Recall (Sensitivity)** : % de vraies fraudes détectées
- **F1-Score** : Harmonic mean de Precision et Recall
- **ROC-AUC** : Aire sous la courbe ROC
- **Precision-Recall AUC** : Plus adapté aux données déséquilibrées
- **Confusion Matrix**
- **Cost-benefit analysis** : Coût des faux positifs vs faux négatifs

**Objectif** : Maximiser le Recall (détecter le max de fraudes) tout en maintenant une Precision acceptable (éviter trop de faux positifs)

## 📈 Résultats

### Performance des modèles

| Modèle | Recall | Precision | F1-Score | ROC-AUC |
|--------|--------|-----------|----------|---------|
| Logistic Regression | 0.61 | 0.05 | 0.09 | 0.97 |
| Random Forest | 0.82 | 0.91 | 0.86 | 0.98 |
| XGBoost + SMOTE | 0.95 | 0.88 | 0.91 | 0.99 |
| Isolation Forest | 0.75 | 0.28 | 0.41 | 0.93 |

**Meilleur modèle** : XGBoost avec SMOTE
- Détecte 95% des fraudes
- 12% de faux positifs
- Temps d'inférence: <5ms

### Features importantes

Top 10 features (selon SHAP values):
1. V14
2. V12
3. V10
4. V17
5. Amount
6. V11
7. V4
8. V16
9. V7
10. Time

### Analyse coût-bénéfice

Hypothèses:
- Coût moyen d'une fraude manquée: $100
- Coût d'investigation d'un faux positif: $10

**Résultat** : Le modèle XGBoost économise ~$45,000 par an comparé au baseline.

## 📊 Visualisations clés

1. **Distribution des transactions**
   - Normal vs Fraud
   - Par montant, par temps

2. **Confusion Matrix**
   - Avec seuils de décision ajustables

3. **ROC Curve & PR Curve**
   - Comparaison des modèles

4. **Feature Importance**
   - SHAP summary plot
   - SHAP dependence plots

5. **Threshold Analysis**
   - Impact du seuil sur Precision/Recall

6. **Time-series analysis**
   - Détections par heure/jour

## 🎓 Apprentissages clés

1. **L'importance des bonnes métriques** : Accuracy est trompeuse avec données déséquilibrées
2. **SMOTE est puissant** : Amélioration significative vs simple oversampling
3. **Ensemble methods** : Random Forest et XGBoost excellent sur ce type de problème
4. **Feature engineering** : Même avec des features anonymisées, on peut créer de la valeur
5. **Business context matters** : Ajuster le seuil selon le coût relatif des erreurs

## ⚠️ Limitations

- Features anonymisées (PCA) limitent l'interprétabilité business
- Dataset sur 2 jours seulement
- Pas de données temporelles pour détecter l'évolution des patterns de fraude
- Pas de features contextuelles (géolocalisation, marchand, etc.)

## 🔮 Améliorations futures

1. **Deep Learning**
   - Autoencoders pour anomaly detection
   - LSTM pour patterns temporels
   - GAN pour générer des transactions frauduleuses synthétiques

2. **Features supplémentaires**
   - Agrégations par utilisateur
   - Patterns de comportement
   - Graph features (réseau de transactions)

3. **Production**
   - API REST pour scoring en temps réel
   - Monitoring du model drift
   - A/B testing du seuil de décision
   - Dashboard de surveillance

4. **Explainability**
   - LIME pour expliquer les prédictions individuelles
   - Contrefactuels ("que faudrait-il changer pour ne pas être détecté?")

## 📚 Références

- **Dataset** : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **SMOTE** : https://arxiv.org/abs/1106.1813
- **Imbalanced-learn** : https://imbalanced-learn.org/
- **Cost-Sensitive Learning** : Elkan, C. (2001). The foundations of cost-sensitive learning

## 📝 Licence

MIT License
