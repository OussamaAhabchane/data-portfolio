# 📉 Customer Churn Prediction & Value Analysis

Prédiction de l'attrition client combinée avec analyse RFM (Recency, Frequency, Monetary) et calcul de la Customer Lifetime Value (CLV).

## 🎯 Objectifs

1. Prédire quels clients risquent de partir (churn)
2. Segmenter les clients avec analyse RFM
3. Calculer la Customer Lifetime Value (CLV)
4. Identifier les leviers d'action pour réduire le churn
5. Prioriser les clients selon leur valeur et risque

## 📊 Dataset

**Source** : Kaggle Telco Customer Churn ou Banking Customer Churn
- https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Caractéristiques** :
- 7,043 clients
- 21 features
- Taux de churn : ~26%

**Variables** :
- **Démographiques** : Gender, SeniorCitizen, Partner, Dependents
- **Services** : PhoneService, InternetService, StreamingTV, etc.
- **Compte** : Tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
- **Target** : Churn (Yes/No)

## 🛠️ Technologies utilisées

```
# Core
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0

# ML
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0

# Visualisation
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Interprétabilité
shap>=0.41.0
lime>=0.2.0

# Utils
imbalanced-learn>=0.9.0
```

## 📁 Structure du projet

```
04-customer-churn-prediction/
├── data/
│   ├── raw/                 # Dataset original
│   ├── processed/           # Features engineered
│   └── download_data.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_rfm_segmentation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_clv_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── rfm.py               # RFM analysis
│   ├── feature_engineering.py
│   ├── models.py
│   ├── clv.py               # CLV calculation
│   └── visualization.py
├── models/
│   └── churn_predictor.pkl
├── requirements.txt
└── README.md
```

## 🚀 Installation et utilisation

### 1. Installation

```bash
cd 04-customer-churn-prediction
pip install -r requirements.txt
```

### 2. Téléchargement des données

```bash
python data/download_data.py
```

### 3. Exécution

```bash
# Notebooks
jupyter notebook notebooks/

# Pipeline complet
python src/main_pipeline.py
```

## 🔍 Méthodologie

### 1. Exploratory Data Analysis (EDA)

**Analyses clés** :
- Distribution du churn par segment
- Tenure vs Churn rate
- Monthly charges vs Churn
- Service adoption patterns
- Corrélations entre features

**Insights recherchés** :
- Quels clients partent le plus ?
- À quel moment (tenure) ?
- Quels services réduisent/augmentent le churn ?
- Impact du pricing

### 2. Segmentation RFM

**RFM Analysis** pour clients transactionnels :

- **Recency** : Dernière interaction (jours)
- **Frequency** : Nombre de transactions
- **Monetary** : Valeur totale dépensée

**Segmentation** :
```
Champions        : RFM = 5-5-5
Loyal Customers  : RFM = 4-5-5
Potential Loyalists : RFM = 3-4-4
At Risk          : RFM = 2-2-3
Can't Lose Them  : RFM = 1-5-5
Lost             : RFM = 1-1-1
```

**Adaptation Telco** :
- R = Tenure (inverse : plus récent = plus long)
- F = Service adoption count
- M = Total Charges

### 3. Feature Engineering

**Features créées** :

A. **Tenure-based**
- Tenure bins (0-12, 12-24, 24-48, 48+)
- Tenure per service
- Churn risk zone (first 6 months high risk)

B. **Monetary**
- Monthly charges bins
- Total charges bins
- Price per service
- Charges growth rate

C. **Service adoption**
- Total services count
- Service diversity score
- Premium services flag
- Bundling score

D. **Contract & Payment**
- Contract type encoding
- Payment method risk score
- Auto-pay flag

E. **Behavioral**
- Service changes (upgrades/downgrades)
- Support tickets count (si disponible)
- Last interaction recency

F. **Interactions**
- Tenure × Monthly Charges
- Service count × Contract type
- Senior × Monthly charges

### 4. Modélisation

**Approche** :
- Binary classification (Churn : Yes/No)
- Imbalanced data (~26% churn)

**Modèles comparés** :
1. **Logistic Regression** (baseline interpretable)
2. **Random Forest**
3. **XGBoost**
4. **LightGBM**
5. **CatBoost** (gère les catégories nativement)

**Techniques d'équilibrage** :
- Class weighting
- SMOTE
- Threshold tuning

**Validation** :
- Stratified K-Fold (5 folds)
- Holdout test set (20%)

**Métriques** :
- **ROC-AUC** : Mesure globale
- **Recall** : Crucial (ne pas manquer des churners)
- **Precision** : Éviter trop de faux positifs (coûts marketing)
- **F1-Score**
- **Profit curve** : Optimiser selon coût d'intervention

**Hyperparameter tuning** :
- Optuna ou GridSearchCV
- Optimisation de Recall sous contrainte de Precision

### 5. Customer Lifetime Value (CLV)

**Formule CLV** :
```
CLV = (Average Monthly Revenue × Gross Margin) × (1 / Churn Rate) × Retention Rate
```

**Ou méthode historique** :
```
CLV = Σ (Revenue_month_i / (1 + discount_rate)^i)
```

**Segments CLV** :
- High Value / Low Churn : Champions (retain !)
- High Value / High Churn : Save them !
- Low Value / High Churn : Let them go
- Low Value / Low Churn : Upsell potential

**Matrice Risque-Valeur** :
```
         Low Risk | High Risk
High CLV    A    |    B      <- Priority 1 & 2
Low CLV     C    |    D
```

### 6. Interprétabilité & Insights

**SHAP (SHapley Additive exPlanations)** :
- Feature importance globale
- Explication par prédiction
- Dependence plots

**LIME** :
- Explication locale
- Pour expliquer prédictions individuelles

**Feature importance** :
- Top 10 features driving churn
- Direction d'impact

## 📈 Résultats attendus

### Performance Modèle

| Modèle | ROC-AUC | Recall | Precision | F1-Score |
|--------|---------|--------|-----------|----------|
| Logistic Reg | 0.82 | 0.71 | 0.58 | 0.64 |
| Random Forest | 0.85 | 0.78 | 0.62 | 0.69 |
| XGBoost | 0.87 | 0.81 | 0.65 | 0.72 |
| LightGBM | 0.88 | 0.83 | 0.67 | 0.74 |

**Meilleur modèle** : LightGBM
- Détecte 83% des churners
- 67% de précision (33% faux positifs)

### Top Features

**Features les plus importantes** (selon SHAP) :
1. **Tenure** (-)
2. **Contract_Month-to-month** (+)
3. **Monthly Charges** (+)
4. **Total Charges** (-)
5. **Internet Service_Fiber optic** (+)
6. **Payment Method_Electronic check** (+)
7. **Tech Support_No** (+)
8. **Online Security_No** (+)

**(+) = augmente le churn, (-) = réduit le churn*

### Segmentation RFM

**Distribution clients** :
- Champions : 12%
- Loyal : 18%
- Potential Loyalists : 22%
- At Risk : 15%
- Can't Lose : 8%
- Lost : 25%

### CLV Analysis

**CLV moyen par segment** :
- Champions : $7,200
- At Risk : $4,800
- Lost : $1,500

**ROI d'une campagne de rétention** :
- Coût intervention : $50/client
- Taux de sauvegarde : 30%
- CLV moyen sauvé : $4,500
- ROI : 2,600%

## 📊 Visualisations clés

1. **Churn Rate by Segment**
   - Tenure bins
   - Contract type
   - Service adoption

2. **Feature Importance**
   - SHAP summary plot
   - SHAP waterfall (explication individuelle)

3. **RFM Segments**
   - Heatmap RFM
   - 3D scatter plot

4. **CLV Distribution**
   - Histogram
   - CLV vs Churn Probability

5. **Risk-Value Matrix**
   - Quadrant plot
   - Bubble size = nombre de clients

6. **ROC & PR Curves**
   - Comparaison des modèles

7. **Confusion Matrix**
   - With optimal threshold

8. **Profit Curve**
   - Expected profit vs threshold

## 💼 Recommandations Business

### Actions par segment

**Segment B (High Value, High Risk)** : 🚨 URGENT
- Contact proactif par account manager
- Offre personnalisée de rétention
- Upgrade gratuit ou discount
- Priorité #1

**Segment A (High Value, Low Risk)** : 💎 NURTURE
- Programme de fidélité
- Early access nouvelles features
- Récompenses

**Segment D (Low Value, High Risk)** : 🤔 ÉVALUER
- Campagne automatisée low-cost
- Laisser partir si coût > bénéfice

**Segment C (Low Value, Low Risk)** : 📈 UPSELL
- Campagne d'upselling
- Bundling de services
- Éducation produit

### Tactiques de réduction du churn

1. **Contractual** : Inciter aux contrats long-terme
2. **Bundling** : Offrir des bundles de services
3. **Support** : Proposer tech support gratuit
4. **Onboarding** : Programme intensif premiers 6 mois
5. **Pricing** : Revoir pricing fiber optic
6. **Payment** : Pousser auto-pay vs check électronique

## 🎓 Apprentissages clés

1. **Tenure matters most** : Les 6 premiers mois sont critiques
2. **Contract = lock-in** : Month-to-month est le plus risqué
3. **Services reduce churn** : Plus de services = plus de stickiness
4. **Price sensitivity exists** : Mais pas le driver principal
5. **Support is crucial** : Tech support réduit significativement le churn

## ⚠️ Limitations

- Dataset Telco spécifique (généralisation limitée)
- Pas de données temporelles fines (pas de séries temporelles)
- Pas d'info sur expérience client (NPS, satisfaction)
- Pas de données compétitives
- Pas de tracking des campagnes de rétention passées

## 🔮 Améliorations futures

1. **Time Series**
   - Séries temporelles de comportement
   - Survival analysis (durée avant churn)
   - Prédiction du timing du churn

2. **Advanced ML**
   - Neural networks
   - AutoML (auto feature engineering)
   - Ensemble stacking

3. **Causal Inference**
   - Uplift modeling (qui répondra aux campagnes ?)
   - Treatment effect estimation
   - A/B testing framework

4. **Real-time**
   - API de scoring en temps réel
   - Dashboard de monitoring
   - Alertes automatiques

5. **Personalization**
   - Recommandation d'offres personnalisées
   - Next Best Action engine
   - Micro-segmentation

6. **External Data**
   - Données compétitives
   - Données macro-économiques
   - Social media sentiment

## 📚 Références

- **CLV** : Fader, P., & Hardie, B. (2013). The Gamma-Gamma Model of Monetary Value
- **RFM** : Hughes, A. M. (1994). Strategic Database Marketing
- **SHAP** : Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions
- **Uplift Modeling** : Rzepakowski, P., & Jaroszewicz, S. (2012). Decision trees for uplift modeling

## 📝 Licence

MIT License
