"""
Data preprocessing module for dealer churn prediction.
Handles feature engineering, encoding, and data preparation for model training.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dlt_utils import DLTReader
from src.model_config import DATA_CONFIG, MODEL_PATHS
from src.utils import validate_dataframe, handle_missing_values
from src.dtype_manager import DtypeManager, load_dataframe_with_dtypes, find_dtype_file


class DataPreprocessor:
    """Handles data preprocessing for dealer churn prediction."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'churn_status'
        
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
    
    def filter_recent_data(self, df):
        """Filter data to keep only recent churners."""
        print("🔍 Filtering data for recent churners...")
        
        original_shape = df.shape
        df = df[df['last_billed_days'] <= DATA_CONFIG["max_last_billed_days"]]
        
        print(f"✅ Filtered data. Original: {original_shape}, Filtered: {df.shape}")
        return df
    
    def create_severity_feature(self, df):
        """Create severity feature based on last billed days."""
        print("📈 Creating severity feature...")
        
        # Calculate severity based on benchmark cutoff
        df['severity'] = round((df['last_billed_days'] / DATA_CONFIG["severity_benchmark_days"]), 4)
        
        print("✅ Severity feature created")
        return df
    
    def encode_dealer_club(self, df):
        """Encode dealer club category using hierarchy mapping."""
        print("🏆 Encoding dealer club categories...")
        
        club_hierarchy = DATA_CONFIG["club_hierarchy"]
        df['dealer_club_category'] = df['dealer_club_category'].apply(
            lambda x: club_hierarchy.get(str(x).strip().upper(), np.nan)
        )
        
        print("✅ Dealer club categories encoded")
        return df
    
    def encode_zone_features(self, df):
        """Create dummy variables for zone features."""
        print("🌍 Encoding zone features...")
        
        zone_dummies = pd.get_dummies(df['zone'], prefix='zone')
        zone_dummies = zone_dummies.astype(int)  # Convert boolean to int
        df = pd.concat([df, zone_dummies], axis=1)
        df.drop("zone", axis=1, inplace=True)
        
        print(f"✅ Zone features encoded. Added {len(zone_dummies.columns)} zone columns")
        return df
    
    def encode_region_features(self, df):
        """Create dummy variables for region features."""
        print("🗺️  Encoding region features...")
        
        region_dummies = pd.get_dummies(df['region_name'], prefix='region_name')
        region_dummies = region_dummies.astype(int)  # Convert boolean to int
        df = pd.concat([df, region_dummies], axis=1)
        df.drop("region_name", axis=1, inplace=True)
        
        print(f"✅ Region features encoded. Added {len(region_dummies.columns)} region columns")
        return df
    
    def encode_territory_features(self, df):
        """Encode territory code using label encoding."""
        print("📍 Encoding territory features...")
        
        le = LabelEncoder()
        df['territory_code_encoded'] = le.fit_transform(df['territory_code'])
        self.label_encoders['territory_code'] = le
        
        df.drop('territory_code', axis=1, inplace=True)
        
        print("✅ Territory features encoded")
        return df
    
    def encode_target_variable(self, df):
        """Encode target variable (churn_status)."""
        print("🎯 Encoding target variable...")
        
        target_encoding = DATA_CONFIG["target_encoding"]
        df[self.target_column] = df[self.target_column].replace(target_encoding)
        
        print("✅ Target variable encoded")
        return df
    
    def handle_categorical_columns(self, df):
        """Handle all remaining categorical columns by converting to numeric."""
        print("🔤 Handling remaining categorical columns...")
        
        # Identify object columns that are not already encoded
        object_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if it's in object columns
        if self.target_column in object_columns:
            object_columns.remove(self.target_column)
        
        # Remove already processed columns
        processed_columns = ['dealer_club_category', 'zone', 'region_name', 'territory_code']
        object_columns = [col for col in object_columns if col not in processed_columns]
        
        if object_columns:
            print(f"   Found {len(object_columns)} categorical columns: {object_columns}")
            
            for col in object_columns:
                try:
                    # Try to convert to numeric first
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"   ✅ Converted {col} to numeric")
                except:
                    # If conversion fails, use label encoding
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"   ✅ Label encoded {col}")
        else:
            print("   ℹ️  No categorical columns to handle")
        
        print("✅ Categorical columns handled")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values and infinite values."""
        print("🔧 Handling missing values...")
        
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill missing values with 0
        df.fillna(0, inplace=True)
        
        print("✅ Missing values handled")
        return df
    
    def remove_unnecessary_columns(self, df):
        """Remove columns that are not needed for modeling."""
        print("🗑️  Removing unnecessary columns...")
        
        columns_to_remove = ['last_billed_days', 'severity']
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        
        if existing_columns:
            df.drop(columns=existing_columns, inplace=True)
            print(f"✅ Removed columns: {existing_columns}")
        else:
            print("ℹ️  No unnecessary columns to remove")
        
        return df
    
    def ensure_numeric_types(self, df):
        """Ensure all columns are numeric types for XGBoost compatibility."""
        print("🔢 Ensuring all columns are numeric...")
        
        # Convert all columns to numeric, coercing errors to NaN
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any NaN values created by conversion
        df.fillna(0, inplace=True)
        
        # Ensure all columns are numeric
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            print(f"   ⚠️  Warning: Still have non-numeric columns: {non_numeric_cols}")
        else:
            print("   ✅ All columns are now numeric")
        
        print("✅ Numeric types ensured")
        return df
    
    def prepare_features_and_target(self, df):
        """Separate features and target variable."""
        print("🎯 Preparing features and target...")
        
        # Ensure all data is numeric before splitting
        df = self.ensure_numeric_types(df)
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        print(f"✅ Features prepared. Shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_data(self, X, y):
        """Split data into training and testing sets."""
        print("✂️  Splitting data into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=DATA_CONFIG["test_size"], 
            stratify=y, 
            random_state=DATA_CONFIG["random_state"]
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training set: {X_train.shape}")
        print(f"   Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_selection(self, X, y, model, threshold_percentile=75):
        """Select top features based on importance threshold."""
        print(f"🔍 Selecting top features based on {100-threshold_percentile}% importance threshold...")
        
        importances = model.feature_importances_
        threshold = np.percentile(importances, threshold_percentile)
        selected_idx = np.where(importances >= threshold)[0]
        selected_features = X.columns[selected_idx]
        
        print(f"✅ Selected {len(selected_features)} features out of {len(X.columns)}")
        return selected_features
    
    def preprocess_data(self, file_path=None):
        """Main preprocessing pipeline."""
        print("🚀 Starting data preprocessing pipeline...")
        print("=" * 50)
        
        try:
            # Load data
            df = self.load_data(file_path)
            
            # Filter recent data
            df = self.filter_recent_data(df)
            
            # Create severity feature (optional, will be removed later)
            df = self.create_severity_feature(df)
            
            # Encode categorical features
            df = self.encode_dealer_club(df)
            df = self.encode_zone_features(df)
            df = self.encode_region_features(df)
            df = self.encode_territory_features(df)
            
            # Encode target variable
            df = self.encode_target_variable(df)
            
            # Handle remaining categorical columns
            df = self.handle_categorical_columns(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Remove unnecessary columns
            df = self.remove_unnecessary_columns(df)
            
            # Prepare features and target
            X, y = self.prepare_features_and_target(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            print("\n🎉 Data preprocessing completed successfully!")
            print(f"Final feature count: {len(self.feature_columns)}")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_columns': self.feature_columns,
                'label_encoders': self.label_encoders
            }
            
        except Exception as e:
            print(f"❌ Error in data preprocessing: {str(e)}")
            raise
    
    def get_preprocessor_info(self):
        """Get information about the preprocessor."""
        return {
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'target_column': self.target_column
        }
