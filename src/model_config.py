"""
Model configuration module for dealer churn prediction.
Contains XGBoost parameters and model settings.
"""

# XGBoost Model Configuration
XGBOOST_CONFIG = {
    # Base parameters
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "scale_pos_weight": None,  # Will be calculated based on class imbalance
    
    # Hyperparameter tuning ranges
    "param_dist": {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'min_child_weight': [1, 3, 5, 7]
    },
    
    # Grid search parameters (refined after random search)
    "param_grid": {
        'n_estimators': [200, 300],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

# Data Processing Configuration
DATA_CONFIG = {
    # Data filtering
    "max_last_billed_days": 365,
    "test_size": 0.2,
    "random_state": 42,
    
    # Feature engineering
    "severity_benchmark_days": 120,
    "feature_selection_threshold": 0.75,  # Top 25% features by importance
    
    # Encoding
    "club_hierarchy": {
        'NS': 0,
        'STARTER': 1,
        'BLUE CLUB': 2,
        'GOLD PLUS CLUB': 3,
        'PLATINUM CLUB': 4,
        'DIAMOND CLUB': 5,
        'ACER CLUB': 6,
        "CHAIRMAN'S CLUB": 7,
        "CHAIRMAN ADVISORY": 8
    },
    
    # Target encoding
    "target_encoding": {
        "Active": 0,
        "Churned": 1
    }
}

# Model Evaluation Configuration
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "scoring_metric": "f1",
    "n_iter_random_search": 20,
    "confidence_threshold": 0.0001,
    
    # Metrics to track
    "metrics": [
        "accuracy",
        "precision",
        "recall", 
        "f1_score",
        "roc_auc",
        "confusion_matrix"
    ]
}

# File Paths
MODEL_PATHS = {
    "input_data": "offset_features.csv",
    "model_output": "models/xgb_churn_model.pkl",
    "preprocessor_output": "models/preprocessor.pkl",
    "feature_importance": "models/feature_importance.csv",
    "predictions_output": "output/predictions.csv",
    "explanations_output": "output/shap_explanations.csv"
}

# SHAP Configuration
SHAP_CONFIG = {
    "max_display_features": 20,
    "top_features_count": 10,
    "explanation_type": "waterfall",  # waterfall, bar, summary
    "save_explanations": True
}
