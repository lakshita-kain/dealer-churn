"""
Production prediction module for dealer churn prediction.
Handles batch predictions for all dealers in production environment.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from src.model_config import MODEL_PATHS, DATA_CONFIG, EVALUATION_CONFIG
from src.data_preprocessor import DataPreprocessor
from src.utils import validate_dataframe, handle_missing_values
from src.dtype_manager import DtypeManager, load_dataframe_with_dtypes, find_dtype_file
from dlt_utils import DLTReader

class ModelPredictor:
    """Handles production predictions for dealer churn."""
    
    def __init__(self, model_path=None, preprocessor_info=None):
        """Initialize the model predictor."""
        self.model_path = model_path or MODEL_PATHS["model_output"]
        self.model = None
        self.preprocessor_info = preprocessor_info
        self.predictions = None
    
    def load_data(self, file_path=None):
        """Load the feature dataset with preserved data types."""
        if file_path is None:
            file_path = MODEL_PATHS["input_data"]
        
        # Check if it's a CSV file with dtype information
        if file_path.endswith('.csv'):
            try:
                # Try to load with preserved data types
                df = load_dataframe_with_dtypes(file_path)
                print(f"✅ Data loaded with preserved data types from {file_path}")
            except FileNotFoundError:
                # Fallback to regular CSV loading
                print(f"📊 Loading data from {file_path} (no dtype file found)...")
                df = pd.read_csv(file_path)
                print(f"⚠️  Warning: Data types may not be preserved")
        else:
            # Load from DLT
            dlt_reader = DLTReader(catalog="provisioned-tableau-data", schema="data_science")
            print(f"📊 Loading data from DLT table: {file_path}...")
            df = dlt_reader.read_table(file_path).toPandas()
            # Try to find and load corresponding dtype file
            dtype_file_path = find_dtype_file(file_path)
            
            if dtype_file_path:
                try:
                    print(f"🔍 Found dtype file: {dtype_file_path}")
                    dtype_manager = DtypeManager(dtype_file_path)
                    df = dtype_manager.restore_dtypes(df)
                    print(f"✅ Data types restored from {dtype_file_path}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not restore data types from {dtype_file_path}: {e}")
            else:
                print(f"⚠️  Warning: No dtype file found for DLT table {file_path}")
                print(f"   This may cause XGBoost compatibility issues if data types are not numeric")
        
        # Set index if dealer_code column exists
        if 'dealer_code' in df.columns:
            df.set_index('dealer_code', inplace=True)
        
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        print(f"📊 Data types summary:")
        print(f"   - Object columns: {len(df.select_dtypes(include=['object']).columns)}")
        print(f"   - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        
        return df

    def load_model(self, model_path=None):
        """Load the trained model and preprocessor info."""
        if model_path is None:
            model_path = self.model_path
        
        print(f"📂 Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Try to load preprocessor info if available
        self.load_preprocessor_info_if_available()
        
        return self.model
    
    def load_preprocessor_info_if_available(self):
        """Try to load preprocessor info from saved files."""
        try:
            # Look for preprocessor info file
            preprocessor_info_path = MODEL_PATHS["model_output"].replace('.pkl', '_preprocessor_info.pkl')
            
            if os.path.exists(preprocessor_info_path):
                self.preprocessor_info = joblib.load(preprocessor_info_path)
                print("✅ Preprocessor info loaded from file")
            else:
                print("⚠️  Warning: Preprocessor info file not found")
                print(f"   Expected location: {preprocessor_info_path}")
                print("   Will need to create preprocessor info from training data")
                
                # Try to create preprocessor info from training data
                self.create_preprocessor_info_from_training_data()
                
        except Exception as e:
            print(f"⚠️  Warning: Could not load preprocessor info: {e}")
            print("   Will need to create preprocessor info from training data")
            self.create_preprocessor_info_from_training_data()
    
    def create_preprocessor_info_from_training_data(self):
        """Create preprocessor info by running preprocessing on training data."""
        try:
            print("🔧 Creating preprocessor info from training data...")
            
            # Use the data preprocessor to create info
            preprocessor = DataPreprocessor()
            
            # Load training data
            training_data = preprocessor.load_data()
            
            # Run preprocessing to get the info
            preprocessed_data = preprocessor.preprocess_data()
            
            # Extract preprocessor info
            self.preprocessor_info = preprocessor.get_preprocessor_info()
            
            print("✅ Preprocessor info created from training data")
            
        except Exception as e:
            print(f"❌ Error creating preprocessor info: {e}")
            print("   You may need to run training first to create preprocessor info")
            self.preprocessor_info = None
    
    def load_preprocessor_info(self, preprocessor_info):
        """Load preprocessor information for consistent data processing."""
        self.preprocessor_info = preprocessor_info
        print("✅ Preprocessor info loaded")
    
    def prepare_production_data(self, df, preprocessor_info=None):
        """Prepare data for production prediction using same preprocessing steps."""
        print("🔧 Preparing data for production prediction...")
        
        if preprocessor_info is None:
            preprocessor_info = self.preprocessor_info
        
        if preprocessor_info is None:
            raise ValueError("Preprocessor info is required for data preparation")
        
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Set dealer_code as index if not already
        if 'dealer_code' in df_processed.columns:
            df_processed.set_index('dealer_code', inplace=True)
        
        # Apply same preprocessing steps as training
        print("📊 Applying preprocessing steps...")
        
        # Filter recent data (same as training)
        df_processed = df_processed[df_processed['last_billed_days'] <= DATA_CONFIG["max_last_billed_days"]]
        
        # Encode dealer club
        club_hierarchy = DATA_CONFIG["club_hierarchy"]
        df_processed['dealer_club_category'] = df_processed['dealer_club_category'].apply(
            lambda x: club_hierarchy.get(str(x).strip().upper(), np.nan)
        )
        
        # Encode zone features
        if 'zone' in df_processed.columns:
            zone_dummies = pd.get_dummies(df_processed['zone'], prefix='zone')
            zone_dummies = zone_dummies.astype(int)
            df_processed = pd.concat([df_processed, zone_dummies], axis=1)
            df_processed.drop("zone", axis=1, inplace=True)
        
        # Encode region features
        if 'region_name' in df_processed.columns:
            region_dummies = pd.get_dummies(df_processed['region_name'], prefix='region_name')
            region_dummies = region_dummies.astype(int)
            df_processed = pd.concat([df_processed, region_dummies], axis=1)
            df_processed.drop("region_name", axis=1, inplace=True)
        
        # Encode territory features
        if 'territory_code' in df_processed.columns and 'territory_code' in preprocessor_info.get('label_encoders', {}):
            le = preprocessor_info['label_encoders']['territory_code']
            df_processed['territory_code_encoded'] = le.transform(df_processed['territory_code'])
            df_processed.drop('territory_code', axis=1, inplace=True)
        
        # Remove unnecessary columns
        columns_to_remove = ['last_billed_days', 'severity', 'churn_status']
        existing_columns = [col for col in columns_to_remove if col in df_processed.columns]
        if existing_columns:
            df_processed.drop(columns=existing_columns, inplace=True)
        
        # Handle missing values
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_processed.fillna(0, inplace=True)
        
        # Ensure feature columns match training data
        expected_features = preprocessor_info.get('feature_columns', [])
        if expected_features:
            # Add missing columns with zeros
            missing_features = set(expected_features) - set(df_processed.columns)
            for feature in missing_features:
                df_processed[feature] = 0
            
            # Reorder columns to match training data
            df_processed = df_processed[expected_features]
        
        print(f"✅ Data prepared. Shape: {df_processed.shape}")
        return df_processed
    
    def predict_churn(self, X, return_probabilities=True):
        """Make churn predictions for given data."""
        if self.model is None:
            self.load_model()
        
        print("🎯 Making churn predictions...")
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        if return_probabilities:
            y_proba = self.model.predict_proba(X)[:, 1]  # Probability of churn
        else:
            y_proba = None
        
        print(f"✅ Predictions completed for {len(X)} dealers")
        return y_pred, y_proba
    
    def create_prediction_results(self, X, y_pred, y_proba, include_features=False):
        """Create comprehensive prediction results DataFrame."""
        print("📋 Creating prediction results...")
        
        # Create base results
        results = pd.DataFrame({
            'dealer_code': X.index,
            'predicted_churn': y_pred,
            'churn_probability': y_proba
        })
        
        # Apply confidence threshold
        threshold = EVALUATION_CONFIG["confidence_threshold"]
        results['churn_probability'] = results['churn_probability'].apply(
            lambda x: 0 if x < threshold else x
        )
        results['churn_probability'] = results['churn_probability'].round(5)
        
        # Add prediction labels
        results['prediction_label'] = results['predicted_churn'].map({0: 'Active', 1: 'Churned'})
        
        # Add risk categories based on probability
        results['risk_category'] = pd.cut(
            results['churn_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        
        # Add prediction timestamp
        results['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Include original features if requested
        if include_features:
            feature_data = X.copy()
            feature_data.index.name = 'dealer_code'
            feature_data = feature_data.reset_index()
            results = pd.merge(results, feature_data, on='dealer_code', how='left')
        
        print("✅ Prediction results created")
        return results
    
    def predict_all_dealers(self, data_source=None, save_results=True, include_features=False):
        """Predict churn for all dealers in the dataset."""
        print("🚀 Starting prediction for all dealers...")
        print("=" * 50)
        
        try:
            # Load data
            if data_source is None:
                data_source = MODEL_PATHS["input_data"]
            
            print(f"📊 Loading data from {data_source}...")
            df = self.load_data(data_source)
            # df = pd.read_csv(data_source)
            
            # Prepare data
            X = self.prepare_production_data(df)
            
            # Make predictions
            y_pred, y_proba = self.predict_churn(X, return_probabilities=True)
            
            # Create results
            results = self.create_prediction_results(X, y_pred, y_proba, include_features)
            
            # Store predictions
            self.predictions = results
            
            # Save results if requested
            if save_results:
                output_path = MODEL_PATHS["predictions_output"]
                self.save_predictions(results, output_path)
            
            # Print summary
            self.print_prediction_summary(results)
            
            print("\n🎉 Prediction completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")
            raise
    
    def predict_specific_dealers(self, dealer_codes, data_source=None, save_results=True):
        """Predict churn for specific dealers."""
        print(f"🎯 Predicting churn for {len(dealer_codes)} specific dealers...")
        
        try:
            # Load data
            if data_source is None:
                data_source = MODEL_PATHS["input_data"]
            
            df = pd.read_csv(data_source)
            
            # Filter for specific dealers
            df_filtered = df[df['dealer_code'].isin(dealer_codes)]
            
            if len(df_filtered) == 0:
                print("⚠️  No dealers found with the provided codes")
                return None
            
            # Prepare data
            X = self.prepare_production_data(df_filtered)
            
            # Make predictions
            y_pred, y_proba = self.predict_churn(X, return_probabilities=True)
            
            # Create results
            results = self.create_prediction_results(X, y_pred, y_proba, include_features=True)
            
            # Save results if requested
            if save_results:
                output_path = f"output/predictions_specific_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.save_predictions(results, output_path)
            
            print("✅ Specific dealer predictions completed!")
            return results
            
        except Exception as e:
            print(f"❌ Error in specific dealer prediction: {str(e)}")
            raise
    
    def save_predictions(self, predictions, file_path=None):
        """Save predictions to CSV file."""
        if file_path is None:
            file_path = MODEL_PATHS["predictions_output"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        print(f"💾 Saving predictions to {file_path}...")
        predictions.to_csv(file_path, index=False)
        
        print("✅ Predictions saved successfully!")
        return file_path
    
    def print_prediction_summary(self, results):
        """Print summary of predictions."""
        print("\n📊 Prediction Summary:")
        print("-" * 30)
        
        # Overall statistics
        total_dealers = len(results)
        predicted_churn = results['predicted_churn'].sum()
        churn_rate = (predicted_churn / total_dealers) * 100
        
        print(f"Total Dealers: {total_dealers}")
        print(f"Predicted to Churn: {predicted_churn} ({churn_rate:.2f}%)")
        print(f"Predicted to Stay Active: {total_dealers - predicted_churn} ({100 - churn_rate:.2f}%)")
        
        # Risk category distribution
        print(f"\nRisk Category Distribution:")
        risk_dist = results['risk_category'].value_counts()
        for category, count in risk_dist.items():
            percentage = (count / total_dealers) * 100
            print(f"  {category}: {count} ({percentage:.2f}%)")
        
        # Probability statistics
        print(f"\nChurn Probability Statistics:")
        prob_stats = results['churn_probability'].describe()
        print(f"  Mean: {prob_stats['mean']:.4f}")
        print(f"  Median: {prob_stats['50%']:.4f}")
        print(f"  Max: {prob_stats['max']:.4f}")
        print(f"  Min: {prob_stats['min']:.4f}")
    
    def get_high_risk_dealers(self, results=None, risk_threshold=0.7):
        """Get dealers with high churn risk."""
        if results is None:
            results = self.predictions
        
        if results is None:
            raise ValueError("No predictions available. Run prediction first.")
        
        high_risk = results[results['churn_probability'] >= risk_threshold].copy()
        high_risk = high_risk.sort_values('churn_probability', ascending=False)
        
        print(f"🚨 Found {len(high_risk)} high-risk dealers (probability >= {risk_threshold})")
        return high_risk
    
    def get_dealer_prediction(self, dealer_code, results=None):
        """Get prediction for a specific dealer."""
        if results is None:
            results = self.predictions
        
        if results is None:
            raise ValueError("No predictions available. Run prediction first.")
        
        dealer_pred = results[results['dealer_code'] == dealer_code]
        
        if len(dealer_pred) == 0:
            print(f"⚠️  No prediction found for dealer {dealer_code}")
            return None
        
        return dealer_pred.iloc[0]
