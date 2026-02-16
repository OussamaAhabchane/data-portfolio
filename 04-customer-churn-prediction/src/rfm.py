"""
RFM (Recency, Frequency, Monetary) Analysis pour segmentation client
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict


class RFMAnalyzer:
    """
    Classe pour effectuer une analyse RFM et segmenter les clients
    """
    
    def __init__(self, df: pd.DataFrame, 
                 customer_id_col: str = 'CustomerID',
                 date_col: str = 'InvoiceDate', 
                 amount_col: str = 'TotalCharges'):
        """
        Args:
            df: DataFrame avec transactions
            customer_id_col: Nom de la colonne client ID
            date_col: Nom de la colonne date
            amount_col: Nom de la colonne montant
        """
        self.df = df.copy()
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.rfm_df = None
        self.segments = None
    
    def calculate_rfm(self, reference_date: datetime = None) -> pd.DataFrame:
        """
        Calcule les scores RFM pour chaque client
        
        Args:
            reference_date: Date de référence (par défaut : max date + 1 jour)
            
        Returns:
            DataFrame avec scores RFM
        """
        print("📊 Calculating RFM scores...")
        
        # Date de référence
        if reference_date is None:
            reference_date = self.df[self.date_col].max() + pd.Timedelta(days=1)
        
        # Calculer R, F, M
        rfm = self.df.groupby(self.customer_id_col).agg({
            self.date_col: lambda x: (reference_date - x.max()).days,  # Recency
            self.customer_id_col: 'count',  # Frequency
            self.amount_col: 'sum'  # Monetary
        })
        
        # Renommer les colonnes
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Réinitialiser l'index
        rfm = rfm.reset_index()
        
        print(f"  ✓ RFM calculated for {len(rfm)} customers")
        print(f"\nRFM Statistics:")
        print(rfm[['Recency', 'Frequency', 'Monetary']].describe())
        
        self.rfm_df = rfm
        return rfm
    
    def score_rfm(self, quartiles: bool = True) -> pd.DataFrame:
        """
        Attribue des scores 1-5 (ou 1-4 pour quartiles) pour R, F, M
        
        Args:
            quartiles: Utiliser des quartiles (4 bins) ou quintiles (5 bins)
            
        Returns:
            DataFrame avec scores RFM
        """
        print("\n🔢 Scoring RFM...")
        
        if self.rfm_df is None:
            self.calculate_rfm()
        
        rfm = self.rfm_df.copy()
        n_bins = 4 if quartiles else 5
        
        # Score Recency (inverse : plus récent = meilleur score)
        rfm['R_Score'] = pd.qcut(
            rfm['Recency'], 
            q=n_bins, 
            labels=list(range(n_bins, 0, -1)),  # Inverse
            duplicates='drop'
        )
        
        # Score Frequency
        rfm['F_Score'] = pd.qcut(
            rfm['Frequency'], 
            q=n_bins, 
            labels=list(range(1, n_bins + 1)),
            duplicates='drop'
        )
        
        # Score Monetary
        rfm['M_Score'] = pd.qcut(
            rfm['Monetary'], 
            q=n_bins, 
            labels=list(range(1, n_bins + 1)),
            duplicates='drop'
        )
        
        # Convertir en int
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Score RFM combiné (string)
        rfm['RFM_Score'] = (
            rfm['R_Score'].astype(str) + 
            rfm['F_Score'].astype(str) + 
            rfm['M_Score'].astype(str)
        )
        
        # Score total (moyenne)
        rfm['RFM_Total'] = rfm[['R_Score', 'F_Score', 'M_Score']].mean(axis=1)
        
        print(f"  ✓ RFM scores calculated")
        
        self.rfm_df = rfm
        return rfm
    
    def segment_customers(self) -> pd.DataFrame:
        """
        Segmente les clients en groupes basés sur les scores RFM
        
        Returns:
            DataFrame avec segments
        """
        print("\n🎯 Segmenting customers...")
        
        if 'R_Score' not in self.rfm_df.columns:
            self.score_rfm()
        
        rfm = self.rfm_df.copy()
        
        def assign_segment(row):
            """Assigne un segment basé sur les scores RFM"""
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            # Champions : RFM tous élevés
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            
            # Loyal Customers : R élevé, F et M moyens-élevés
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            
            # Potential Loyalists : R élevé, F et M moyens
            elif r >= 3 and f >= 2 and m >= 2:
                return 'Potential Loyalists'
            
            # Recent Customers : R élevé, F et M bas
            elif r >= 4 and f <= 2:
                return 'Recent Customers'
            
            # Promising : R et M moyens, F bas
            elif r >= 3 and m >= 2 and f <= 2:
                return 'Promising'
            
            # Need Attention : R moyen, F et M moyens
            elif r == 3 and f >= 2 and m >= 2:
                return 'Need Attention'
            
            # About to Sleep : R moyen-bas, F et M moyens
            elif r == 2 and f >= 2 and m >= 2:
                return 'About to Sleep'
            
            # At Risk : R bas, F et M élevés (clients précieux qui s'éloignent)
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'
            
            # Can't Lose Them : R très bas, F et M très élevés
            elif r == 1 and f >= 4 and m >= 4:
                return "Can't Lose Them"
            
            # Hibernating : R bas, F moyen, M moyen
            elif r <= 2 and f <= 3 and m <= 3:
                return 'Hibernating'
            
            # Lost : RFM tous bas
            elif r == 1 and f == 1:
                return 'Lost'
            
            # Autres
            else:
                return 'Others'
        
        # Appliquer la segmentation
        rfm['Segment'] = rfm.apply(assign_segment, axis=1)
        
        # Statistiques par segment
        segment_stats = rfm.groupby('Segment').agg({
            self.customer_id_col: 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'RFM_Total': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Avg_RFM_Score']
        segment_stats = segment_stats.sort_values('Avg_RFM_Score', ascending=False)
        
        print("\n📊 Segment Distribution:")
        print(segment_stats)
        
        self.rfm_df = rfm
        self.segments = segment_stats
        
        return rfm
    
    def visualize_rfm(self, save_path: str = None):
        """
        Crée des visualisations RFM
        
        Args:
            save_path: Chemin pour sauvegarder les plots
        """
        if self.rfm_df is None or 'Segment' not in self.rfm_df.columns:
            self.segment_customers()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribution des segments
        segment_counts = self.rfm_df['Segment'].value_counts()
        axes[0, 0].bar(range(len(segment_counts)), segment_counts.values)
        axes[0, 0].set_xticks(range(len(segment_counts)))
        axes[0, 0].set_xticklabels(segment_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('Customer Segments Distribution')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # 2. RFM Score Distribution
        axes[0, 1].hist(self.rfm_df['RFM_Total'], bins=30, edgecolor='black')
        axes[0, 1].set_title('RFM Score Distribution')
        axes[0, 1].set_xlabel('RFM Total Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Recency vs Frequency
        for segment in self.rfm_df['Segment'].unique():
            data = self.rfm_df[self.rfm_df['Segment'] == segment]
            axes[0, 2].scatter(data['Recency'], data['Frequency'], 
                             alpha=0.5, label=segment, s=50)
        axes[0, 2].set_xlabel('Recency (days)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Recency vs Frequency by Segment')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 4. Monetary Distribution by Segment
        segment_order = self.segments.index
        rfm_sorted = self.rfm_df.set_index('Segment').loc[segment_order].reset_index()
        axes[1, 0].boxplot(
            [rfm_sorted[rfm_sorted['Segment'] == seg]['Monetary'].values 
             for seg in segment_order],
            labels=segment_order
        )
        axes[1, 0].set_xticklabels(segment_order, rotation=45, ha='right')
        axes[1, 0].set_title('Monetary Value by Segment')
        axes[1, 0].set_ylabel('Monetary Value')
        
        # 5. Heatmap RFM moyenne par segment
        heatmap_data = self.rfm_df.groupby('Segment')[['R_Score', 'F_Score', 'M_Score']].mean()
        heatmap_data = heatmap_data.loc[segment_order]
        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlGnBu', 
                   ax=axes[1, 1], cbar_kws={'label': 'Average Score'})
        axes[1, 1].set_title('Average RFM Scores by Segment')
        axes[1, 1].set_xlabel('Segment')
        axes[1, 1].set_ylabel('RFM Component')
        
        # 6. 3D Scatter (projection 2D)
        for segment in self.rfm_df['Segment'].unique():
            data = self.rfm_df[self.rfm_df['Segment'] == segment]
            axes[1, 2].scatter(
                data['Frequency'], 
                data['Monetary'], 
                s=100/data['Recency'],  # Size inversement proportionnel à Recency
                alpha=0.5, 
                label=segment
            )
        axes[1, 2].set_xlabel('Frequency')
        axes[1, 2].set_ylabel('Monetary')
        axes[1, 2].set_title('Frequency vs Monetary (bubble size = Recency⁻¹)')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plots saved to {save_path}")
        
        plt.show()
    
    def get_segment_recommendations(self) -> Dict[str, str]:
        """
        Retourne des recommandations d'action par segment
        
        Returns:
            Dict {segment: recommandation}
        """
        recommendations = {
            'Champions': '🌟 Reward them! Make them brand ambassadors. Ask for reviews.',
            'Loyal Customers': '💎 Upsell higher value products. Engage regularly.',
            'Potential Loyalists': '📈 Offer membership. Recommend products.',
            'Recent Customers': '🎯 Provide onboarding support. Build relationship.',
            'Promising': '💰 Offer free shipping or discounts on next purchase.',
            'Need Attention': '🔔 Make limited time offers. Recommend products.',
            'About to Sleep': '⏰ Reactivation campaign. Special offers.',
            'At Risk': '🚨 Win them back via renewals or newer products.',
            "Can't Lose Them": '❗ Win them back. Survey them. Make special offers.',
            'Hibernating': '💤 Recreate brand value. Reactivate with promotions.',
            'Lost': '⚰️ Consider cost-benefit of reactivation vs. let go.',
            'Others': '❓ Analyze further and segment more granularly.'
        }
        return recommendations


# Exemple d'utilisation pour Telco Churn (adaptation RFM)
def adapt_rfm_for_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapte RFM pour un dataset Telco où il n'y a pas de transactions multiples
    
    Args:
        df: DataFrame Telco avec tenure, services, charges
        
    Returns:
        DataFrame avec RFM adapté
    """
    rfm_telco = df.copy()
    
    # Recency → Inverse de Tenure (clients récents ont tenure faible)
    rfm_telco['Recency'] = df['tenure'].max() - df['tenure']
    
    # Frequency → Nombre de services adoptés
    service_cols = [col for col in df.columns if 'Service' in col or col in 
                   ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']]
    rfm_telco['Frequency'] = df[service_cols].apply(
        lambda row: sum([1 for val in row if val == 'Yes']), axis=1
    )
    
    # Monetary → TotalCharges
    rfm_telco['Monetary'] = df['TotalCharges']
    
    return rfm_telco[['customerID', 'Recency', 'Frequency', 'Monetary']]


if __name__ == "__main__":
    print("RFM Analyzer module loaded successfully!")
    print("\nUsage example:")
    print("  analyzer = RFMAnalyzer(df, 'CustomerID', 'Date', 'Amount')")
    print("  rfm = analyzer.calculate_rfm()")
    print("  rfm_scored = analyzer.score_rfm()")
    print("  rfm_segmented = analyzer.segment_customers()")
    print("  analyzer.visualize_rfm('rfm_analysis.png')")
