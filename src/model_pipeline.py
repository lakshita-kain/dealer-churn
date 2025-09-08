"""
Main pipeline orchestrator for dealer churn prediction.
Handles complete training and prediction workflows.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from .data_preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_predictor import ModelPredictor
from .model_explainer import ModelExplainer
from .model_config import MODEL_PATHS
from .utils import print_pipeline_progress


class ModelPipeline:
    """Main pipeline orchestrator for dealer churn prediction."""
    
    def __init__(self):
        """Initialize the model pipeline."""
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.predictor = ModelPredictor(preprocessor_info=self.preprocessor.get_preprocessor_info())
        self.explainer = ModelExplainer()
        
        self.training_results = None
        self.evaluation_results = None
        self.prediction_results = None
        self.explanation_results = None
        
    def run_complete_training_pipeline(self, data_path=None, use_hyperparameter_tuning=True, 
                                     save_model=True, evaluate_model=True, explain_model=True):
        """Run complete training pipeline with evaluation and explanation."""
        print("🚀 Starting Complete Training Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Data Preprocessing
            print_pipeline_progress("Data Preprocessing", 1, 5, "Preparing and encoding data")
            preprocessed_data = self.preprocessor.preprocess_data(data_path)
            
            # Step 2: Model Training
            print_pipeline_progress("Model Training", 2, 5, "Training XGBoost model")
            if use_hyperparameter_tuning:
                training_results = self.trainer.train_with_hyperparameter_tuning(
                    preprocessed_data['X_train'],
                    preprocessed_data['y_train'],
                    preprocessed_data['X_test'],
                    preprocessed_data['y_test']
                )
            else:
                training_results = self.trainer.train_simple_model(
                    preprocessed_data['X_train'],
                    preprocessed_data['y_train']
                )
            
            self.training_results = training_results
            
            # Step 3: Model Evaluation
            if evaluate_model:
                print_pipeline_progress("Model Evaluation", 3, 5, "Evaluating model performance")
                evaluation_results = self.evaluator.evaluate_model(
                    training_results['model'],
                    preprocessed_data['X_train'],
                    preprocessed_data['y_train'],
                    preprocessed_data['X_test'],
                    preprocessed_data['y_test'],
                    plot_results=True,
                    save_results=True
                )
                self.evaluation_results = evaluation_results
            
            # Step 4: Model Explanation
            if explain_model:
                print_pipeline_progress("Model Explanation", 4, 5, "Generating SHAP explanations")
                explanation_results = self.explainer.explain_model_comprehensively(
                    training_results['model'],
                    preprocessed_data['X_train'],
                    preprocessed_data['X_test'],
                    preprocessed_data['y_test'],
                    save_explanations=True,
                    plot_results=True
                )
                self.explanation_results = explanation_results
            
            # Step 5: Setup Predictor
            print_pipeline_progress("Pipeline Setup", 5, 5, "Setting up production predictor")
            
            # Get preprocessor info
            preprocessor_info = self.preprocessor.get_preprocessor_info()
            
            # Save preprocessor info with the model
            self.trainer.save_model(
                model=training_results['model'],
                file_path=training_results['model_path'],
                preprocessor_info=preprocessor_info
            )
            
            # Setup predictor
            self.predictor.load_model(training_results['model_path'])
            self.predictor.load_preprocessor_info(preprocessor_info)
            
            print("\n🎉 Complete training pipeline finished successfully!")
            print(f"Model saved to: {training_results['model_path']}")
            print(f"Feature importance saved to: {training_results['importance_path']}")
            
            return {
                'preprocessed_data': preprocessed_data,
                'training_results': training_results,
                'evaluation_results': self.evaluation_results,
                'explanation_results': self.explanation_results,
                'predictor': self.predictor
            }
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {str(e)}")
            raise
    
    def run_production_prediction_pipeline(self, data_path=None, save_results=True):
        """Run production prediction pipeline for all dealers."""
        print("🚀 Starting Production Prediction Pipeline")
        print("=" * 50)
        
        try:
            # Load model and preprocessor info
            self.predictor.load_model()
            # Run predictions for all dealers
            prediction_results = self.predictor.predict_all_dealers(
                data_source=data_path,
                save_results=save_results,
                include_features=False
            )
            
            self.prediction_results = prediction_results
            
            print("\n🎉 Production prediction pipeline completed successfully!")
            return prediction_results
            
        except Exception as e:
            print(f"❌ Error in prediction pipeline: {str(e)}")
            raise
    
    def run_specific_dealer_prediction(self, dealer_codes, data_path=None, save_results=True):
        """Run prediction for specific dealers."""
        print(f"🎯 Starting Specific Dealer Prediction for {len(dealer_codes)} dealers")
        print("=" * 50)
        
        try:
            # Load model and preprocessor info
            self.predictor.load_model()
            
            # Run predictions for specific dealers
            prediction_results = self.predictor.predict_specific_dealers(
                dealer_codes=dealer_codes,
                data_source=data_path,
                save_results=save_results
            )
            
            print("\n🎉 Specific dealer prediction completed successfully!")
            return prediction_results
            
        except Exception as e:
            print(f"❌ Error in specific dealer prediction: {str(e)}")
            raise
    
    def get_high_risk_dealers(self, risk_threshold=0.7):
        """Get high-risk dealers from latest predictions."""
        if self.prediction_results is None:
            print("⚠️  No prediction results available. Run prediction pipeline first.")
            return None
        
        return self.predictor.get_high_risk_dealers(self.prediction_results, risk_threshold)
    
    def explain_dealer_prediction(self, dealer_code):
        """Explain prediction for a specific dealer."""
        if self.explanation_results is None:
            print("⚠️  No explanation results available. Run training pipeline with explanation first.")
            return None
        
        return self.explainer.get_dealer_explanation(dealer_code, None, self.explanation_results['explanations_df'])
    
    def get_model_performance_summary(self):
        """Get summary of model performance."""
        if self.evaluation_results is None:
            print("⚠️  No evaluation results available. Run training pipeline with evaluation first.")
            return None
        
        test_metrics = self.evaluation_results['test_metrics']
        
        summary = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'roc_auc': test_metrics.get('roc_auc', 'N/A')
        }
        
        print("📊 Model Performance Summary:")
        print("-" * 30)
        for metric, value in summary.items():
            print(f"{metric.replace('_', ' ').title()}: {value}")
        
        return summary
    
    def get_prediction_summary(self):
        """Get summary of latest predictions."""
        if self.prediction_results is None:
            print("⚠️  No prediction results available. Run prediction pipeline first.")
            return None
        
        self.predictor.print_prediction_summary(self.prediction_results)
        return self.prediction_results
    
    def save_pipeline_results(self, output_dir="output"):
        """Save all pipeline results to files."""
        print(f"💾 Saving pipeline results to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save training results
        if self.training_results:
            training_file = os.path.join(output_dir, f"training_results_{timestamp}.csv")
            if 'importance_df' in self.training_results:
                self.training_results['importance_df'].to_csv(training_file, index=False)
        
        # Save evaluation results
        if self.evaluation_results:
            evaluation_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
            self.evaluator.save_evaluation_results(evaluation_file)
        
        # Save prediction results
        if self.prediction_results is not None:
            prediction_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")
            self.predictor.save_predictions(self.prediction_results, prediction_file)
        
        # Save explanation results
        if self.explanation_results:
            explanation_file = os.path.join(output_dir, f"explanations_{timestamp}.csv")
            self.explainer.save_explanations(self.explanation_results['explanations_df'], explanation_file)
        
        print("✅ Pipeline results saved successfully!")
    
    def run_quick_training(self, data_path=None):
        """Run quick training without hyperparameter tuning."""
        print("⚡ Starting Quick Training Pipeline")
        print("=" * 40)
        
        try:
            # Data preprocessing
            preprocessed_data = self.preprocessor.preprocess_data(data_path)
            
            # Simple model training
            training_results = self.trainer.train_simple_model(
                preprocessed_data['X_train'],
                preprocessed_data['y_train']
            )
            
            # Basic evaluation
            evaluation_results = self.evaluator.evaluate_model(
                training_results['model'],
                preprocessed_data['X_train'],
                preprocessed_data['y_train'],
                preprocessed_data['X_test'],
                preprocessed_data['y_test'],
                plot_results=False,
                save_results=False
            )
            
            # Get preprocessor info
            preprocessor_info = self.preprocessor.get_preprocessor_info()
            
            # Save preprocessor info with the model
            self.trainer.save_model(
                model=training_results['model'],
                file_path=training_results['model_path'],
                preprocessor_info=preprocessor_info
            )
            
            # Setup predictor
            self.predictor.load_model(training_results['model_path'])
            self.predictor.load_preprocessor_info(preprocessor_info)
            
            print("\n🎉 Quick training completed successfully!")
            
            return {
                'preprocessed_data': preprocessed_data,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'predictor': self.predictor
            }
            
        except Exception as e:
            print(f"❌ Error in quick training: {str(e)}")
            raise


def run_complete_pipeline(data_path=None, use_hyperparameter_tuning=True):
    """Convenience function to run the complete pipeline."""
    pipeline = ModelPipeline()
    return pipeline.run_complete_training_pipeline(
        data_path=data_path,
        use_hyperparameter_tuning=use_hyperparameter_tuning
    )


def run_production_predictions(data_path=None):
    """Convenience function to run production predictions."""
    pipeline = ModelPipeline()
    return pipeline.run_production_prediction_pipeline(data_path=data_path)


if __name__ == "__main__":
    # Example usage
    pipeline = ModelPipeline()
    
    # Run complete training pipeline
    results = pipeline.run_complete_training_pipeline()
    
    # Run production predictions
    predictions = pipeline.run_production_prediction_pipeline()
    
    # Get high-risk dealers
    high_risk = pipeline.get_high_risk_dealers()
    
    print("🎉 Pipeline execution completed!")
