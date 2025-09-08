"""
Model training module for dealer churn prediction.
Handles XGBoost model training with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
from .model_config import XGBOOST_CONFIG, EVALUATION_CONFIG, MODEL_PATHS
from .utils import print_pipeline_progress


class ModelTrainer:
    """Handles XGBoost model training and hyperparameter optimization."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.model = None
        self.best_model = None
        self.training_history = {}
        
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced dataset."""
        print("⚖️  Calculating class weights...")
        
        class_counts = y.value_counts()
        total_samples = len(y)
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = (total_samples - class_counts[1]) / class_counts[1]
        
        print(f"✅ Class distribution: {dict(class_counts)}")
        print(f"✅ Scale pos weight: {scale_pos_weight:.4f}")
        
        return scale_pos_weight
    
    def create_base_model(self, y_train):
        """Create base XGBoost model with calculated class weights."""
        print("🏗️  Creating base XGBoost model...")
        
        # Calculate class weights
        scale_pos_weight = self.calculate_class_weights(y_train)
        
        # Create model with base parameters
        model_params = XGBOOST_CONFIG.copy()
        model_params["scale_pos_weight"] = scale_pos_weight
        
        # Remove parameter tuning configs
        model_params.pop("param_dist", None)
        model_params.pop("param_grid", None)
        
        self.model = XGBClassifier(**model_params)
        
        print("✅ Base XGBoost model created")
        return self.model
    
    def perform_random_search(self, X_train, y_train):
        """Perform randomized search for hyperparameter tuning."""
        print("🎲 Performing randomized search...")
        
        # Create base model
        base_model = self.create_base_model(y_train)
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=XGBOOST_CONFIG["param_dist"],
            n_iter=EVALUATION_CONFIG["n_iter_random_search"],
            scoring=EVALUATION_CONFIG["scoring_metric"],
            cv=EVALUATION_CONFIG["cv_folds"],
            verbose=1,
            n_jobs=-1,
            random_state=XGBOOST_CONFIG["random_state"]
        )
        
        print("🔄 Starting randomized search...")
        random_search.fit(X_train, y_train)
        
        print(f"✅ Randomized search completed!")
        print(f"   Best params: {random_search.best_params_}")
        print(f"   Best score: {random_search.best_score_:.4f}")
        
        self.training_history['random_search'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_
        }
        
        return random_search
    
    def perform_grid_search(self, X_train, y_train, random_search_results=None):
        """Perform grid search for refined hyperparameter tuning."""
        print("🔍 Performing grid search...")
        
        # Create model with best params from random search or base params
        if random_search_results:
            best_params = random_search_results.best_params_
            # Update base model with best params
            base_model = XGBClassifier(**best_params)
        else:
            base_model = self.create_base_model(y_train)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=XGBOOST_CONFIG["param_grid"],
            scoring=EVALUATION_CONFIG["scoring_metric"],
            cv=EVALUATION_CONFIG["cv_folds"],
            verbose=1,
            n_jobs=-1
        )
        
        print("🔄 Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Grid search completed!")
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   Best score: {grid_search.best_score_:.4f}")
        
        self.training_history['grid_search'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        return grid_search
    
    def train_final_model(self, X_train, y_train, best_params=None):
        """Train the final model with best parameters."""
        print("🎯 Training final model...")
        
        if best_params:
            # Use provided best parameters
            model_params = XGBOOST_CONFIG.copy()
            model_params.update(best_params)
            model_params["scale_pos_weight"] = self.calculate_class_weights(y_train)
            
            # Remove parameter tuning configs
            model_params.pop("param_dist", None)
            model_params.pop("param_grid", None)
            
            self.best_model = XGBClassifier(**model_params)
        else:
            # Use base model
            self.best_model = self.create_base_model(y_train)
        
        # Train the model
        self.best_model.fit(X_train, y_train)
        
        print("✅ Final model trained successfully!")
        return self.best_model
    
    def save_model(self, model=None, file_path=None, preprocessor_info=None):
        """Save the trained model and preprocessor info."""
        if model is None:
            model = self.best_model
            
        if file_path is None:
            file_path = MODEL_PATHS["model_output"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        print(f"💾 Saving model to {file_path}...")
        joblib.dump(model, file_path)
        
        # Save preprocessor info if provided
        if preprocessor_info is not None:
            preprocessor_info_path = file_path.replace('.pkl', '_preprocessor_info.pkl')
            print(f"💾 Saving preprocessor info to {preprocessor_info_path}...")
            joblib.dump(preprocessor_info, preprocessor_info_path)
            print("✅ Preprocessor info saved!")
        
        print("✅ Model saved successfully!")
        return file_path
    
    def load_model(self, file_path=None):
        """Load a saved model."""
        if file_path is None:
            file_path = MODEL_PATHS["model_output"]
        
        print(f"📂 Loading model from {file_path}...")
        model = joblib.load(file_path)
        
        print("✅ Model loaded successfully!")
        return model
    
    def get_feature_importance(self, model=None, feature_names=None):
        """Get feature importance from the trained model."""
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("No model available. Train a model first.")
        
        print("📊 Extracting feature importance...")
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("✅ Feature importance extracted")
        return importance_df
    
    def save_feature_importance(self, importance_df, file_path=None):
        """Save feature importance to CSV."""
        if file_path is None:
            file_path = MODEL_PATHS["feature_importance"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        print(f"💾 Saving feature importance to {file_path}...")
        importance_df.to_csv(file_path, index=False)
        
        print("✅ Feature importance saved!")
        return file_path
    
    def train_with_hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        """Complete training pipeline with hyperparameter tuning."""
        print("🚀 Starting model training with hyperparameter tuning...")
        print("=" * 60)
        
        try:
            # Step 1: Randomized Search
            print_pipeline_progress("Randomized Search", 1, 3, "Finding initial best parameters")
            random_search = self.perform_random_search(X_train, y_train)
            
            # Step 2: Grid Search
            print_pipeline_progress("Grid Search", 2, 3, "Refining parameters")
            grid_search = self.perform_grid_search(X_train, y_train, random_search)
            
            # Step 3: Train Final Model
            print_pipeline_progress("Final Training", 3, 3, "Training with best parameters")
            final_model = self.train_final_model(X_train, y_train, grid_search.best_params_)
            
            # Step 4: Save Model
            model_path = self.save_model(final_model)
            
            # Step 5: Extract and Save Feature Importance
            importance_df = self.get_feature_importance(final_model, X_train.columns)
            importance_path = self.save_feature_importance(importance_df)
            
            print("\n🎉 Model training completed successfully!")
            print(f"Model saved to: {model_path}")
            print(f"Feature importance saved to: {importance_path}")
            
            return {
                'model': final_model,
                'model_path': model_path,
                'importance_df': importance_df,
                'importance_path': importance_path,
                'training_history': self.training_history
            }
            
        except Exception as e:
            print(f"❌ Error in model training: {str(e)}")
            raise
    
    def train_simple_model(self, X_train, y_train):
        """Train a simple model without hyperparameter tuning."""
        print("🚀 Starting simple model training...")
        
        try:
            # Train final model with base parameters
            final_model = self.train_final_model(X_train, y_train)
            
            # Save model
            model_path = self.save_model(final_model)
            
            # Extract and save feature importance
            importance_df = self.get_feature_importance(final_model, X_train.columns)
            importance_path = self.save_feature_importance(importance_df)
            
            print("\n🎉 Simple model training completed successfully!")
            print(f"Model saved to: {model_path}")
            print(f"Feature importance saved to: {importance_path}")
            
            return {
                'model': final_model,
                'model_path': model_path,
                'importance_df': importance_df,
                'importance_path': importance_path
            }
            
        except Exception as e:
            print(f"❌ Error in simple model training: {str(e)}")
            raise
