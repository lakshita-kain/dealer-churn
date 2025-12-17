"""
Model explainability module for dealer churn prediction.
Handles SHAP explanations and model interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from datetime import datetime
from .model_config import SHAP_CONFIG, MODEL_PATHS


class ModelExplainer:
    """Handles model explainability using SHAP values."""
    
    def __init__(self, model=None, X_train=None):
        """Initialize the model explainer."""
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        self.explanations = None
        
    def create_explainer(self, model=None, X_train=None):
        """Create SHAP explainer for the model."""
        if model is None:
            model = self.model
        if X_train is None:
            X_train = self.X_train
        
        if model is None or X_train is None:
            raise ValueError("Model and training data are required to create explainer")
        
        print("🔍 Creating SHAP explainer...")
        
        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(model, X_train)
        
        print("✅ SHAP explainer created successfully!")
        return self.explainer
    
    def calculate_shap_values(self, X, explainer=None):
        """Calculate SHAP values for given data."""
        if explainer is None:
            explainer = self.explainer
        
        if explainer is None:
            raise ValueError("SHAP explainer not found. Create explainer first.")
        
        print(f"🧮 Calculating SHAP values for {len(X)} samples...")
        
        # Calculate SHAP values
        self.shap_values = explainer(X)
        
        print("✅ SHAP values calculated successfully!")
        return self.shap_values
    
    def plot_global_importance(self, shap_values=None, max_display=None, figsize=(10, 8)):
        """Plot global feature importance using SHAP values."""
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not found. Calculate SHAP values first.")
        
        if max_display is None:
            max_display = SHAP_CONFIG["max_display_features"]
        
        print(f"📊 Plotting global feature importance (top {max_display})...")
        
        # Plot global importance
        plt.figure(figsize=figsize)
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.title(f'Global Feature Importance (Top {max_display})')
        plt.tight_layout()
        plt.show()
        
        print("✅ Global importance plot created")
    
    def plot_summary_plot(self, shap_values=None, X=None, max_display=None, figsize=(10, 8)):
        """Plot SHAP summary plot."""
        if shap_values is None:
            shap_values = self.shap_values
        if X is None:
            X = self.X_train
        
        if shap_values is None or X is None:
            raise ValueError("SHAP values and feature data are required")
        
        if max_display is None:
            max_display = SHAP_CONFIG["max_display_features"]
        
        print(f"📊 Plotting SHAP summary plot (top {max_display})...")
        
        # Plot summary
        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title(f'SHAP Summary Plot (Top {max_display})')
        plt.tight_layout()
        plt.show()
        
        print("✅ Summary plot created")
    
    def explain_individual_prediction(self, shap_values=None, X=None, index=0, 
                                    max_display=None, figsize=(10, 8)):
        """Explain individual prediction using waterfall plot."""
        if shap_values is None:
            shap_values = self.shap_values
        if X is None:
            X = self.X_train
        
        if shap_values is None or X is None:
            raise ValueError("SHAP values and feature data are required")
        
        if max_display is None:
            max_display = SHAP_CONFIG["max_display_features"]
        
        print(f"🔍 Explaining individual prediction for index {index}...")
        
        # Create explanation object
        exp = shap.Explanation(
            values=shap_values[index],
            base_values=self.explainer.expected_value,
            data=X.iloc[index].values,
            feature_names=X.columns
        )
        
        # Plot waterfall
        plt.figure(figsize=figsize)
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        plt.title(f'Individual Prediction Explanation (Index {index})')
        plt.tight_layout()
        plt.show()
        
        print("✅ Individual explanation created")
    
    def get_top_features_for_dealer(self, dealer_code, X, shap_values=None, top_n=None):
        """Get top contributing features for a specific dealer."""
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not found. Calculate SHAP values first.")
        
        if top_n is None:
            top_n = SHAP_CONFIG["top_features_count"]
        
        # Find dealer index
        if dealer_code in X.index:
            dealer_idx = X.index.get_loc(dealer_code)
        else:
            raise ValueError(f"Dealer {dealer_code} not found in data")
        
        # Get SHAP values for this dealer
        dealer_shap = shap_values[dealer_idx]
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'shap_value': dealer_shap,
            'feature_value': X.iloc[dealer_idx].values
        })
        
        # Sort by absolute SHAP value
        feature_importance['abs_shap_value'] = np.abs(feature_importance['shap_value'])
        feature_importance = feature_importance.sort_values('abs_shap_value', ascending=False)
        
        # Get top features
        top_features = feature_importance.head(top_n)
        
        print(f"✅ Top {top_n} features for dealer {dealer_code}")
        return top_features
    
    def create_explanations_dataframe(self, X, y_pred=None, shap_values=None):
        """Create comprehensive explanations DataFrame."""
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not found. Calculate SHAP values first.")
        
        print("📋 Creating explanations DataFrame...")
        
        # Convert SHAP values to DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
        
        # Create results list
        results = []
        from tqdm import tqdm
        for i, dealer_code in tqdm(enumerate(X.index)):
            row_shap = shap_df.loc[dealer_code]
            
            # Get positive (churn-increasing) features
            positive_features = row_shap[row_shap > 0].sort_values(ascending=False)
            
            # Get negative (churn-decreasing) features
            negative_features = row_shap[row_shap < 0].sort_values(ascending=True)
            
            # Get top features by absolute value
            abs_features = row_shap.abs().sort_values(ascending=False)
            
            results.append({
                'dealer_code': dealer_code,
                'predicted_churn': y_pred[dealer_code] if y_pred is not None else None,
                'positive_features': dict(positive_features),
                'negative_features': dict(negative_features),
                'top_features': dict(abs_features.head(SHAP_CONFIG["top_features_count"])),
                'total_positive_impact': positive_features.sum(),
                'total_negative_impact': negative_features.sum(),
                'net_impact': row_shap.sum()
            })
        
        explanations_df = pd.DataFrame(results)
        explanations_df.set_index('dealer_code', inplace=True)
        
        print("✅ Explanations DataFrame created")
        return explanations_df
    
    def save_explanations(self, explanations_df, file_path=None):
        """Save explanations to CSV file."""
        if file_path is None:
            file_path = MODEL_PATHS["explanations_output"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        print(f"💾 Saving explanations to {file_path}...")
        
        # Convert dictionaries to strings for CSV compatibility
        explanations_export = explanations_df.copy()
        
        for col in ['positive_features', 'negative_features', 'top_features']:
            if col in explanations_export.columns:
                explanations_export[col] = explanations_export[col].astype(str)
        
        explanations_export.to_csv(file_path)
        
        print("✅ Explanations saved successfully!")
        return file_path
    
    def analyze_churn_drivers(self, explanations_df, top_n=10):
        """Analyze most common churn drivers across all dealers."""
        print(f"🔍 Analyzing top {top_n} churn drivers...")
        
        # Collect all positive features
        all_positive_features = {}
        
        for idx, row in explanations_df.iterrows():
            positive_features = eval(row['positive_features']) if isinstance(row['positive_features'], str) else row['positive_features']
            for feature, value in positive_features.items():
                if feature not in all_positive_features:
                    all_positive_features[feature] = []
                all_positive_features[feature].append(value)
        
        # Calculate statistics for each feature
        feature_stats = {}
        for feature, values in all_positive_features.items():
            feature_stats[feature] = {
                'count': len(values),
                'mean_impact': np.mean(values),
                'max_impact': np.max(values),
                'total_impact': np.sum(values)
            }
        
        # Create DataFrame and sort by total impact
        churn_drivers_df = pd.DataFrame(feature_stats).T
        churn_drivers_df = churn_drivers_df.sort_values('total_impact', ascending=False)
        
        print(f"✅ Top {top_n} churn drivers identified")
        return churn_drivers_df.head(top_n)
    
    def explain_model_comprehensively(self, model, X_train, X_test, y_test=None, 
                                    save_explanations=True, plot_results=True):
        """Comprehensive model explanation pipeline."""
        print("🚀 Starting comprehensive model explanation...")
        print("=" * 50)
        
        try:
            # Set model and training data
            self.model = model
            self.X_train = X_train
            
            # Create explainer
            self.create_explainer(model, X_train)
            
            # Calculate SHAP values for test set
            self.calculate_shap_values(X_test)
            
            # Create explanations DataFrame
            explanations_df = self.create_explanations_dataframe(X_test, y_test)
            
            # Save explanations if requested
            if save_explanations:
                self.save_explanations(explanations_df)
            
            # Generate plots if requested
            if plot_results:
                print("\n📊 Generating explanation plots...")
                
                # Global importance
                self.plot_global_importance()
                
                # Summary plot
                self.plot_summary_plot(X=X_test)
                
                # Individual explanations for a few examples
                for i in range(min(3, len(X_test))):
                    self.explain_individual_prediction(X=X_test, index=i)
            
            # Analyze churn drivers
            churn_drivers = self.analyze_churn_drivers(explanations_df)
            
            print("\n🎉 Model explanation completed successfully!")
            
            return {
                'explanations_df': explanations_df,
                'churn_drivers': churn_drivers,
                'explainer': self.explainer,
                'shap_values': self.shap_values
            }
            
        except Exception as e:
            print(f"❌ Error in model explanation: {str(e)}")
            raise
    
    def get_dealer_explanation(self, dealer_code, X, explanations_df=None):
        """Get detailed explanation for a specific dealer."""
        if explanations_df is None:
            explanations_df = self.explanations
        
        if explanations_df is None:
            raise ValueError("No explanations available. Run comprehensive explanation first.")
        
        if dealer_code not in explanations_df.index:
            print(f"⚠️  No explanation found for dealer {dealer_code}")
            return None
        
        dealer_explanation = explanations_df.loc[dealer_code]
        
        print(f"🔍 Explanation for Dealer {dealer_code}:")
        print("-" * 40)
        print(f"Predicted Churn: {dealer_explanation['predicted_churn']}")
        print(f"Net Impact: {dealer_explanation['net_impact']:.4f}")
        print(f"Total Positive Impact: {dealer_explanation['total_positive_impact']:.4f}")
        print(f"Total Negative Impact: {dealer_explanation['total_negative_impact']:.4f}")
        
        # Parse and display top features
        top_features = eval(dealer_explanation['top_features']) if isinstance(dealer_explanation['top_features'], str) else dealer_explanation['top_features']
        
        print(f"\nTop Contributing Features:")
        for feature, impact in list(top_features.items())[:10]:
            print(f"  {feature}: {impact:.4f}")
        
        return dealer_explanation
