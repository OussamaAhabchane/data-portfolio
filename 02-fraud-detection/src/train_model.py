"""
Script principal pour entraîner et évaluer des modèles de détection de fraude
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve,
    roc_curve, f1_score, average_precision_score
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionPipeline:
    """Pipeline complet pour la détection de fraude"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Chemin vers le fichier CSV
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
    
    def load_data(self):
        """Charge les données"""
        print("📊 Chargement des données...")
        self.df = pd.read_csv(self.data_path)
        print(f"  Shape: {self.df.shape}")
        print(f"  Fraudes: {self.df['Class'].sum()} ({self.df['Class'].mean()*100:.3f}%)")
        return self
    
    def preprocess(self):
        """Preprocessing des données"""
        print("\n🔧 Preprocessing...")
        
        # Séparer features et target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        # Normaliser Amount et Time
        if 'Amount' in X.columns:
            X['Amount_log'] = np.log1p(X['Amount'])
        
        if 'Time' in X.columns:
            # Extraire l'heure du jour (cyclique)
            X['Hour'] = (X['Time'] / 3600) % 24
            X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
            X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
        
        # Train-test split AVANT le resampling (important!)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisation
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
        self.X_train[numeric_cols] = self.scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])
        
        print(f"  Train: {self.X_train.shape}")
        print(f"  Test: {self.X_test.shape}")
        print(f"  Fraudes train: {self.y_train.sum()} ({self.y_train.mean()*100:.3f}%)")
        
        return self
    
    def apply_sampling(self, method: str = 'smote'):
        """
        Applique une technique de sampling
        
        Args:
            method: 'none', 'undersample', 'oversample', 'smote', 'adasyn', 'smote_tomek'
        """
        print(f"\n⚖️  Applying {method} sampling...")
        
        if method == 'none':
            return self.X_train, self.y_train
        
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(self.X_train, self.y_train)
        
        elif method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(self.X_train, self.y_train)
        
        elif method == 'smote':
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_res, y_res = smote.fit_resample(self.X_train, self.y_train)
        
        elif method == 'adasyn':
            adasyn = ADASYN(random_state=42)
            X_res, y_res = adasyn.fit_resample(self.X_train, self.y_train)
        
        elif method == 'smote_tomek':
            smt = SMOTETomek(random_state=42)
            X_res, y_res = smt.fit_resample(self.X_train, self.y_train)
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        print(f"  Original: {len(self.y_train)} samples, {self.y_train.sum()} frauds")
        print(f"  Resampled: {len(y_res)} samples, {y_res.sum()} frauds")
        
        return X_res, y_res
    
    def train_model(self, model_type: str = 'xgboost', 
                   sampling: str = 'smote', **model_params):
        """
        Entraîne un modèle
        
        Args:
            model_type: 'logistic', 'random_forest', 'xgboost', 'isolation_forest'
            sampling: Méthode de sampling à appliquer
            **model_params: Paramètres du modèle
        """
        print(f"\n🤖 Training {model_type} with {sampling} sampling...")
        
        # Appliquer le sampling
        X_res, y_res = self.apply_sampling(sampling)
        
        # Initialiser le modèle
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                **model_params
            )
        
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == 'xgboost':
            # Calculer scale_pos_weight pour XGBoost
            scale_pos_weight = (len(y_res) - y_res.sum()) / y_res.sum()
            
            self.model = xgb.XGBClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.1),
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='aucpr'
            )
        
        elif model_type == 'isolation_forest':
            contamination = self.y_train.mean()  # Proportion de fraudes
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Entraîner
        self.model.fit(X_res, y_res)
        print("  ✓ Model trained!")
        
        return self
    
    def evaluate(self, threshold: float = 0.5):
        """
        Évalue le modèle
        
        Args:
            threshold: Seuil de décision
        """
        print(f"\n📊 Evaluation (threshold={threshold})...")
        
        # Prédictions
        if isinstance(self.model, IsolationForest):
            # Isolation Forest retourne -1 pour anomalies
            y_pred = (self.model.predict(self.X_test) == -1).astype(int)
            y_proba = self.model.score_samples(self.X_test)
        else:
            y_proba = self.model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        
        # Métriques
        print("\n" + "="*60)
        print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTrue Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        # Scores
        if not isinstance(self.model, IsolationForest):
            auc_roc = roc_auc_score(self.y_test, y_proba)
            auc_pr = average_precision_score(self.y_test, y_proba)
            print(f"\nROC-AUC: {auc_roc:.4f}")
            print(f"PR-AUC: {auc_pr:.4f}")
        
        f1 = f1_score(self.y_test, y_pred)
        print(f"F1-Score: {f1:.4f}")
        
        # Stocker les résultats
        self.results = {
            'predictions': y_pred,
            'probabilities': y_proba if not isinstance(self.model, IsolationForest) else None,
            'confusion_matrix': cm,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
        
        return self
    
    def plot_results(self, save_path: str = None):
        """Génère des visualisations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        if self.results['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(self.y_test, self.results['probabilities'])
            auc = roc_auc_score(self.y_test, self.results['probabilities'])
            axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        if self.results['probabilities'] is not None:
            precision, recall, _ = precision_recall_curve(self.y_test, self.results['probabilities'])
            auc_pr = average_precision_score(self.y_test, self.results['probabilities'])
            axes[1, 0].plot(recall, precision, label=f'PR (AUC={auc_pr:.3f})')
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision-Recall Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance (si disponible)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20
            
            axes[1, 1].barh(range(len(indices)), importances[indices])
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([self.X_train.columns[i] for i in indices])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 20 Feature Importances')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plots saved to {save_path}")
        
        plt.show()
        
        return self
    
    def save_model(self, path: str):
        """Sauvegarde le modèle"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
        print(f"✓ Model saved to {path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer le pipeline
    pipeline = FraudDetectionPipeline('data/raw/creditcard.csv')
    
    # Charger et préprocesser
    pipeline.load_data().preprocess()
    
    # Entraîner avec XGBoost et SMOTE
    pipeline.train_model(
        model_type='xgboost',
        sampling='smote',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    # Évaluer
    pipeline.evaluate(threshold=0.5)
    
    # Visualiser
    pipeline.plot_results(save_path='results/fraud_detection_results.png')
    
    # Sauvegarder
    pipeline.save_model('models/xgboost_fraud_detector.pkl')
