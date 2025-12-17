"""
Utilities module for dealer churn analysis.
Contains helper functions and data validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str = "DataFrame") -> bool:
    """Validate that a DataFrame has required columns."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"❌ {name} is missing required columns: {missing_columns}")
        return False
    
    print(f"✅ {name} validation passed")
    return True


def check_data_quality(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, any]:
    """Check data quality metrics for a DataFrame."""
    quality_report = {
        'name': name,
        'shape': df.shape,
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        'duplicates': df.duplicated().sum()
    }
    
    print(f"📊 {name} Quality Report:")
    print(f"   Shape: {quality_report['shape']}")
    print(f"   Memory Usage: {quality_report['memory_usage']:.2f} MB")
    print(f"   Duplicates: {quality_report['duplicates']}")
    
    return quality_report


def safe_merge(left_df: pd.DataFrame, right_df: pd.DataFrame, 
               on: str, how: str = 'left', 
               left_name: str = "Left DataFrame", 
               right_name: str = "Right DataFrame") -> pd.DataFrame:
    """Safely merge two DataFrames with validation."""
    print(f"🔄 Merging {left_name} with {right_name} on '{on}' using {how} join...")
    
    # Check if merge columns exist
    if on not in left_df.columns:
        raise ValueError(f"Column '{on}' not found in {left_name}")
    if on not in right_df.columns:
        raise ValueError(f"Column '{on}' not found in {right_name}")
    
    # Perform merge
    merged_df = pd.merge(left_df, right_df, on=on, how=how)
    
    print(f"✅ Merge completed. Result shape: {merged_df.shape}")
    return merged_df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Handle missing values in DataFrame."""
    df_copy = df.copy()
    
    if strategy == 'drop':
        if columns:
            df_copy = df_copy.dropna(subset=columns)
        else:
            df_copy = df_copy.dropna()
        print(f"🗑️  Dropped rows with missing values. New shape: {df_copy.shape}")
    
    elif strategy == 'fill':
        if columns:
            for col in columns:
                if df_copy[col].dtype in ['int64', 'float64']:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                else:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else 'Unknown')
        print(f"🔧 Filled missing values in specified columns")
    
    return df_copy


def convert_date_columns(df: pd.DataFrame, date_columns: List[str], 
                        format: str = None) -> pd.DataFrame:
    """Convert specified columns to datetime format."""
    df_copy = df.copy()
    
    for col in date_columns:
        if col in df_copy.columns:
            try:
                if format:
                    df_copy[col] = pd.to_datetime(df_copy[col], format=format, errors='coerce')
                else:
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                print(f"📅 Converted column '{col}' to datetime")
            except Exception as e:
                print(f"⚠️  Warning: Could not convert column '{col}' to datetime: {e}")
    
    return df_copy


def create_summary_statistics(df: pd.DataFrame, group_by: str, 
                            numeric_columns: List[str]) -> pd.DataFrame:
    """Create summary statistics for numeric columns grouped by a categorical column."""
    summary = df.groupby(group_by)[numeric_columns].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary = summary.reset_index()
    
    return summary


def detect_outliers(df: pd.DataFrame, columns: List[str], 
                   method: str = 'iqr', threshold: float = 1.5) -> Dict[str, List[int]]:
    """Detect outliers in specified columns."""
    outliers = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                outliers[col] = outlier_indices
                
                print(f"🔍 Column '{col}': {len(outlier_indices)} outliers detected")
    
    return outliers


def save_dataframe_info(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame information to a text file."""
    with open(file_path, 'w') as f:
        f.write("DataFrame Information\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n")
        
        f.write("Column Information:\n")
        f.write("-" * 30 + "\n")
        for col in df.columns:
            f.write(f"{col}: {df[col].dtype}\n")
            f.write(f"  Null count: {df[col].isnull().sum()}\n")
            f.write(f"  Unique values: {df[col].nunique()}\n")
            if df[col].dtype in ['int64', 'float64']:
                f.write(f"  Range: {df[col].min()} to {df[col].max()}\n")
            f.write("\n")
    
    print(f"📝 DataFrame information saved to {file_path}")


def print_pipeline_progress(step: str, current: int, total: int, 
                           description: str = "") -> None:
    """Print formatted pipeline progress."""
    progress = (current / total) * 100
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f"\n{step} [{bar}] {progress:.1f}% ({current}/{total})")
    if description:
        print(f"   {description}")


def standardize_period_column(df: pd.DataFrame, column_name: str = 'period') -> pd.DataFrame:
    """Standardize period column to string format for consistent merging."""
    if column_name in df.columns:
        df_copy = df.copy()
        if df_copy[column_name].dtype != 'object':
            print(f"Converting {column_name} column to string format")
            df_copy[column_name] = df_copy[column_name].astype(str)
        return df_copy
    return df
