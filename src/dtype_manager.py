"""
Data type manager for preserving and restoring column data types.
Ensures consistent data types between feature engineering and model training.
"""

import pandas as pd
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class DtypeManager:
    """Manages data type preservation and restoration for DataFrames."""
    
    def __init__(self, dtype_file_path: str = "data_types.json"):
        """
        Initialize the dtype manager.
        
        Args:
            dtype_file_path: Path to save/load data type information
        """
        self.dtype_file_path = dtype_file_path
        self.dtype_info = {}
    
    def save_dtypes(self, df: pd.DataFrame, file_path: Optional[str] = None) -> str:
        """
        Save DataFrame data types to a JSON file.
        
        Args:
            df: DataFrame to extract data types from
            file_path: Optional custom file path for saving
            
        Returns:
            Path to the saved dtype file
        """
        if file_path is None:
            file_path = self.dtype_file_path
        
        # Extract data type information
        dtype_info = {
            'dtypes': df.dtypes.astype(str).to_dict(),
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'index_dtype': str(df.index.dtype) if df.index.dtype else None,
            'has_index_name': df.index.name is not None,
            'index_name': df.index.name
        }
        
        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(dtype_info, f, indent=2)
        
        print(f"✅ Data types saved to: {file_path}")
        print(f"   Columns: {len(dtype_info['columns'])}")
        print(f"   Shape: {dtype_info['shape']}")
        
        return file_path
    
    def load_dtypes(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load data type information from JSON file.
        
        Args:
            file_path: Optional custom file path for loading
            
        Returns:
            Dictionary containing data type information
        """
        if file_path is None:
            file_path = self.dtype_file_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dtype file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            dtype_info = json.load(f)
        
        print(f"✅ Data types loaded from: {file_path}")
        print(f"   Columns: {len(dtype_info['columns'])}")
        print(f"   Shape: {dtype_info['shape']}")
        
        self.dtype_info = dtype_info
        return dtype_info
    
    def restore_dtypes(self, df: pd.DataFrame, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Restore DataFrame data types from saved information.
        
        Args:
            df: DataFrame to restore data types for
            file_path: Optional custom file path for loading dtype info
            
        Returns:
            DataFrame with restored data types
        """
        # Load dtype information if not already loaded
        if not self.dtype_info or file_path:
            self.load_dtypes(file_path)
        
        df_restored = df.copy()
        
        # Restore column data types
        for col, dtype in self.dtype_info['dtypes'].items():
            if col in df_restored.columns:
                try:
                    # Handle special data types
                    if dtype == 'object':
                        df_restored[col] = df_restored[col].astype('object')
                    elif dtype == 'int64':
                        df_restored[col] = pd.to_numeric(df_restored[col], errors='coerce').astype('int64')
                    elif dtype == 'float64':
                        df_restored[col] = pd.to_numeric(df_restored[col], errors='coerce').astype('float64')
                    elif dtype == 'bool':
                        df_restored[col] = df_restored[col].astype('bool')
                    elif dtype.startswith('datetime'):
                        df_restored[col] = pd.to_datetime(df_restored[col], errors='coerce')
                    else:
                        # For other types, try to convert
                        df_restored[col] = df_restored[col].astype(dtype)
                    
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not restore dtype for column '{col}': {e}")
                    # Keep original dtype if conversion fails
                    pass
        
        # Restore index data type if applicable
        if self.dtype_info.get('index_dtype') and self.dtype_info['index_dtype'] != 'None':
            try:
                if self.dtype_info['index_dtype'] == 'object':
                    df_restored.index = df_restored.index.astype('object')
                elif self.dtype_info['index_dtype'] == 'int64':
                    df_restored.index = pd.to_numeric(df_restored.index, errors='coerce').astype('int64')
                # Add more index type handling as needed
            except Exception as e:
                print(f"   ⚠️  Warning: Could not restore index dtype: {e}")
        
        # Restore index name if it existed
        if self.dtype_info.get('has_index_name') and self.dtype_info.get('index_name'):
            df_restored.index.name = self.dtype_info['index_name']
        
        print(f"✅ Data types restored for {len(df_restored.columns)} columns")
        return df_restored
    
    def get_dtype_summary(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of saved data types.
        
        Args:
            file_path: Optional custom file path for loading dtype info
            
        Returns:
            Dictionary with dtype summary information
        """
        if not self.dtype_info or file_path:
            self.load_dtypes(file_path)
        
        # Count data types
        dtype_counts = {}
        for dtype in self.dtype_info['dtypes'].values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        summary = {
            'total_columns': len(self.dtype_info['columns']),
            'shape': self.dtype_info['shape'],
            'dtype_counts': dtype_counts,
            'object_columns': [col for col, dtype in self.dtype_info['dtypes'].items() if dtype == 'object'],
            'numeric_columns': [col for col, dtype in self.dtype_info['dtypes'].items() if dtype in ['int64', 'float64']],
            'datetime_columns': [col for col, dtype in self.dtype_info['dtypes'].items() if dtype.startswith('datetime')]
        }
        
        return summary
    
    def validate_dtypes(self, df: pd.DataFrame, file_path: Optional[str] = None) -> bool:
        """
        Validate that DataFrame data types match saved information.
        
        Args:
            df: DataFrame to validate
            file_path: Optional custom file path for loading dtype info
            
        Returns:
            True if data types match, False otherwise
        """
        if not self.dtype_info or file_path:
            self.load_dtypes(file_path)
        
        mismatches = []
        
        for col, expected_dtype in self.dtype_info['dtypes'].items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    mismatches.append({
                        'column': col,
                        'expected': expected_dtype,
                        'actual': actual_dtype
                    })
        
        if mismatches:
            print("❌ Data type mismatches found:")
            for mismatch in mismatches:
                print(f"   {mismatch['column']}: expected {mismatch['expected']}, got {mismatch['actual']}")
            return False
        else:
            print("✅ All data types match saved information")
            return True


def save_dataframe_with_dtypes(df: pd.DataFrame, csv_path: str, dtype_path: Optional[str] = None) -> tuple:
    """
    Convenience function to save DataFrame with data type information.
    
    Args:
        df: DataFrame to save
        csv_path: Path to save CSV file
        dtype_path: Optional path for dtype file (defaults to csv_path + '_dtypes.json')
        
    Returns:
        Tuple of (csv_path, dtype_path)
    """
    if dtype_path is None:
        dtype_path = csv_path.replace('.csv', '_dtypes.json')
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ DataFrame saved to: {csv_path}")
    
    # Save data types
    dtype_manager = DtypeManager(dtype_path)
    dtype_manager.save_dtypes(df)
    
    return csv_path, dtype_path


def find_dtype_file(base_path: str) -> Optional[str]:
    """
    Find dtype file for a given base path by searching common locations.
    
    Args:
        base_path: Base path (CSV file path or table name)
        
    Returns:
        Path to dtype file if found, None otherwise
    """
    # Remove .csv extension if present
    if base_path.endswith('.csv'):
        base_path = base_path.replace('.csv', '')
    
    # Common possible locations for dtype file
    possible_paths = [
        f"{base_path}_dtypes.json",
        f"offset_features_dtypes.json",  # Common case
        f"data_science/{base_path}_dtypes.json",
        f"data_science/offset_features_dtypes.json",
        f"./{base_path}_dtypes.json",
        f"./offset_features_dtypes.json"
    ]
    
    for dtype_path in possible_paths:
        if os.path.exists(dtype_path):
            return dtype_path
    
    return None


def load_dataframe_with_dtypes(csv_path: str, dtype_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load DataFrame with restored data types.
    
    Args:
        csv_path: Path to CSV file
        dtype_path: Optional path for dtype file (defaults to csv_path + '_dtypes.json')
        
    Returns:
        DataFrame with restored data types
    """
    if dtype_path is None:
        dtype_path = csv_path.replace('.csv', '_dtypes.json')
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✅ DataFrame loaded from: {csv_path}")
    
    # Restore data types
    dtype_manager = DtypeManager(dtype_path)
    df_restored = dtype_manager.restore_dtypes(df)
    
    return df_restored


def load_dataframe_with_dtype_discovery(csv_path: str) -> pd.DataFrame:
    """
    Load DataFrame with automatic dtype file discovery.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with restored data types (if dtype file found)
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✅ DataFrame loaded from: {csv_path}")
    
    # Try to find dtype file
    dtype_path = find_dtype_file(csv_path)
    
    if dtype_path:
        try:
            print(f"🔍 Found dtype file: {dtype_path}")
            dtype_manager = DtypeManager(dtype_path)
            df_restored = dtype_manager.restore_dtypes(df)
            print(f"✅ Data types restored from {dtype_path}")
            return df_restored
        except Exception as e:
            print(f"⚠️  Warning: Could not restore data types from {dtype_path}: {e}")
            return df
    else:
        print(f"⚠️  Warning: No dtype file found for {csv_path}")
        return df


# Example usage
if __name__ == "__main__":
    print("🧪 Testing DtypeManager")
    print("=" * 40)
    
    # Create sample data with mixed types
    sample_data = pd.DataFrame({
        'dealer_code': ['D001', 'D002', 'D003'],
        'dealer_club_category': ['Starter', 'Blue Club', 'Gold Plus Club'],
        'last_billed_days': [30, 45, 60],
        'churn_status': ['Active', 'Churned', 'Active'],
        'numeric_feature': [1.5, 2.3, 3.7],
        'dealer_count': [100, 200, 150]  # This will be object type
    })
    
    print("Original data types:")
    print(sample_data.dtypes)
    
    # Test saving
    csv_path, dtype_path = save_dataframe_with_dtypes(sample_data, "test_data.csv")
    
    # Test loading
    loaded_data = load_dataframe_with_dtypes(csv_path, dtype_path)
    
    print("\nRestored data types:")
    print(loaded_data.dtypes)
    
    # Test validation
    dtype_manager = DtypeManager(dtype_path)
    is_valid = dtype_manager.validate_dtypes(loaded_data)
    
    print(f"\nData type validation: {'✅ Passed' if is_valid else '❌ Failed'}")
    
    # Clean up
    os.remove(csv_path)
    os.remove(dtype_path)
    print("\n✅ Test completed and cleaned up")
