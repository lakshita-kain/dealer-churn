"""
Model evaluation module for dealer churn prediction.
Handles model evaluation, metrics calculation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay
)
from .model_config import EVALUATION_CONFIG


class ModelEvaluator:
    """Handles model evaluation and performance metrics."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive evaluation metrics."""
        print("📊 Calculating evaluation metrics...")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC (if probabilities available)
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        print("✅ Metrics calculated successfully")
        return metrics
    
    def print_metrics(self, metrics, dataset_name="Dataset"):
        """Print formatted metrics."""
        print(f"\n📈 {dataset_name} Performance Metrics:")
        print("-" * 40)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n📋 Classification Report:")
        print("-" * 40)
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", 
                            normalize=True, figsize=(8, 6)):
        """Plot confusion matrix."""
        print(f"📊 Plotting confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=["Active", "Churned"]
        )
        
        if normalize:
            disp.plot(ax=ax, values_format=".2f", colorbar=True)
        else:
            disp.plot(ax=ax, values_format="d", colorbar=True)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        
        print("✅ Confusion matrix plotted")
    
    def plot_roc_curve(self, y_true, y_proba, title="ROC Curve", figsize=(8, 6)):
        """Plot ROC curve."""
        print(f"📊 Plotting ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("✅ ROC curve plotted")
    
    def plot_feature_importance(self, importance_df, top_n=20, figsize=(10, 8)):
        """Plot feature importance."""
        print(f"📊 Plotting top {top_n} feature importance...")
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        print("✅ Feature importance plotted")
    
    def plot_class_distribution(self, y, title="Class Distribution", figsize=(6, 6)):
        """Plot class distribution."""
        print("📊 Plotting class distribution...")
        
        class_counts = pd.Series(y).value_counts()
        labels = ["Active", "Churned"] if set(class_counts.index) == {0, 1} else class_counts.index
        
        # Function to show both percentage and count
        def autopct_format(values):
            def inner_autopct(pct):
                total = sum(values)
                count = int(round(pct * total / 100.0))
                return f"{pct:.1f}%\n({count})"
            return inner_autopct
        
        # Plot
        plt.figure(figsize=figsize)
        class_counts.plot(
            kind='pie',
            labels=labels,
            autopct=autopct_format(class_counts),
            colors=['skyblue', 'salmon'],
            startangle=90
        )
        plt.title(title)
        plt.ylabel("")
        plt.show()
        
        print("✅ Class distribution plotted")
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, 
                      plot_results=True, save_results=False):
        """Comprehensive model evaluation."""
        print("🔍 Starting comprehensive model evaluation...")
        print("=" * 50)
        
        try:
            # Get predictions
            print("🎯 Generating predictions...")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            print("📊 Calculating metrics...")
            train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_proba)
            
            # Print metrics
            self.print_metrics(train_metrics, "Training")
            print(classification_report(y_train, y_train_pred, digits=3))
            
            self.print_metrics(test_metrics, "Testing")
            print(classification_report(y_test, y_test_pred, digits=3))
            
            # Store results
            self.evaluation_results = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'y_train_proba': y_train_proba,
                'y_test_proba': y_test_proba
            }
            
            # Plot results if requested
            if plot_results:
                print("\n📊 Generating visualizations...")
                
                # Class distribution
                self.plot_class_distribution(y_test, "Test Set Class Distribution")
                
                # Confusion matrix
                self.plot_confusion_matrix(y_test, y_test_pred, "Test Set Confusion Matrix")
                
                # ROC curve
                self.plot_roc_curve(y_test, y_test_proba, "Test Set ROC Curve")
            
            # Save results if requested
            if save_results:
                self.save_evaluation_results()
            
            print("\n🎉 Model evaluation completed successfully!")
            
            return self.evaluation_results
            
        except Exception as e:
            print(f"❌ Error in model evaluation: {str(e)}")
            raise
    
    def save_evaluation_results(self, file_path="output/evaluation_results.csv"):
        """Save evaluation results to CSV."""
        print(f"💾 Saving evaluation results to {file_path}...")
        
        # Create results DataFrame
        results_data = []
        
        for dataset in ['train', 'test']:
            metrics = self.evaluation_results[f'{dataset}_metrics']
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':  # Skip confusion matrix
                    results_data.append({
                        'dataset': dataset,
                        'metric': metric,
                        'value': value
                    })
        
        results_df = pd.DataFrame(results_data)
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(file_path, index=False)
        
        print("✅ Evaluation results saved!")
        return file_path
    
    def get_prediction_summary(self, X_test, y_test, y_pred, y_proba):
        """Get summary of predictions with confidence scores."""
        print("📋 Generating prediction summary...")
        
        # Create prediction summary
        prediction_summary = pd.DataFrame({
            'dealer_code': X_test.index,
            'true_label': y_test,
            'predicted_label': y_pred,
            'confidence_score': y_proba
        })
        
        # Apply confidence threshold
        threshold = EVALUATION_CONFIG["confidence_threshold"]
        prediction_summary['confidence_score'] = prediction_summary['confidence_score'].apply(
            lambda x: 0 if x < threshold else x
        )
        prediction_summary['confidence_score'] = prediction_summary['confidence_score'].round(5)
        
        # Add prediction status
        prediction_summary['prediction_correct'] = (
            prediction_summary['true_label'] == prediction_summary['predicted_label']
        )
        
        print("✅ Prediction summary generated")
        return prediction_summary
    
    def analyze_prediction_errors(self, prediction_summary):
        """Analyze prediction errors and misclassifications."""
        print("🔍 Analyzing prediction errors...")
        
        # Get misclassified predictions
        errors = prediction_summary[~prediction_summary['prediction_correct']]
        
        if len(errors) > 0:
            print(f"\n❌ Found {len(errors)} misclassified predictions:")
            print(f"   False Positives (Predicted Churn, Actually Active): {len(errors[errors['predicted_label'] == 1])}")
            print(f"   False Negatives (Predicted Active, Actually Churned): {len(errors[errors['predicted_label'] == 0])}")
            
            # Show confidence distribution for errors
            print(f"\n📊 Confidence score distribution for errors:")
            print(errors['confidence_score'].describe())
        else:
            print("✅ No prediction errors found!")
        
        return errors
