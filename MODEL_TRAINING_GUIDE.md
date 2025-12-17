# Dealer Churn Model Training Guide

This guide shows you how to use the modular model training utilities for dealer churn prediction.

## 🚀 Quick Start

### 1. **Basic Model Training**

```python
from src.model_pipeline import ModelPipeline

# Initialize the pipeline
pipeline = ModelPipeline()

# Quick training (fast, good for testing)
results = pipeline.run_quick_training(data_path="offset_features.csv")

print(f"Model saved to: {results['training_results']['model_path']}")
print(f"Accuracy: {results['training_results']['accuracy']:.3f}")
```

### 2. **Complete Training with Hyperparameter Tuning**

```python
from src.model_pipeline import run_complete_pipeline

# Complete training with hyperparameter optimization
results = run_complete_pipeline(
    data_path="offset_features.csv",
    use_hyperparameter_tuning=True
)

print(f"Best model accuracy: {results['training_results']['accuracy']:.3f}")
print(f"Best parameters: {results['training_results']['best_params']}")
```

### 3. **Production Predictions**

```python
from src.model_pipeline import run_production_predictions

# Predict churn for all dealers
predictions = run_production_predictions(data_path="offset_features.csv")

print(f"Predictions shape: {predictions.shape}")
print(f"High-risk dealers: {len(predictions[predictions['churn_probability'] > 0.7])}")
```

## 📊 Available Methods

### **ModelPipeline Class Methods**

| Method | Description | Use Case |
|--------|-------------|----------|
| `run_quick_training()` | Fast training without hyperparameter tuning | Testing, development |
| `run_complete_training()` | Full training with hyperparameter optimization | Production models |
| `run_production_predictions()` | Predict for all dealers | Production deployment |
| `run_specific_dealer_prediction()` | Predict for specific dealers | Targeted analysis |
| `get_model_performance_summary()` | Get model performance metrics | Model evaluation |
| `get_prediction_summary()` | Get prediction statistics | Results analysis |
| `get_high_risk_dealers()` | Get dealers above risk threshold | Risk management |

### **Individual Module Usage**

#### **Data Preprocessing**
```python
from src.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
data = preprocessor.load_and_preprocess_data("offset_features.csv")
X_train, X_test, y_train, y_test = preprocessor.split_data(data)
```

#### **Model Training**
```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_model(X_train, y_train)
best_model = trainer.hyperparameter_tuning(X_train, y_train)
```

#### **Model Evaluation**
```python
from src.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test)
evaluator.plot_confusion_matrix(results['y_pred'], y_test)
```

#### **Model Predictions**
```python
from src.model_predictor import ModelPredictor

predictor = ModelPredictor()
predictions = predictor.predict_for_dealers(data_path="offset_features.csv")
risk_groups = predictor.categorize_risk_groups(predictions)
```

#### **Model Explanation**
```python
from src.model_explainer import ModelExplainer

explainer = ModelExplainer()
explainer.plot_feature_importance(model, X_train)
explainer.plot_summary_plot(model, X_train)
```

## 🎯 Common Use Cases

### **1. Training a New Model**

```python
from src.model_pipeline import ModelPipeline

pipeline = ModelPipeline()

# For development/testing
results = pipeline.run_quick_training("offset_features.csv")

# For production
results = pipeline.run_complete_training("offset_features.csv")
```

### **2. Making Predictions on New Data**

```python
from src.model_predictor import ModelPredictor

predictor = ModelPredictor()

# Load new data and make predictions
predictions = predictor.predict_for_dealers("new_data.csv")

# Get high-risk dealers
high_risk = predictions[predictions['churn_probability'] > 0.7]
print(f"High-risk dealers: {len(high_risk)}")
```

### **3. Analyzing Model Performance**

```python
from src.model_pipeline import ModelPipeline

pipeline = ModelPipeline()

# Get performance summary
performance = pipeline.get_model_performance_summary()
print(f"Accuracy: {performance['accuracy']:.3f}")
print(f"Precision: {performance['precision']:.3f}")
print(f"Recall: {performance['recall']:.3f}")

# Get high-risk dealers
high_risk = pipeline.get_high_risk_dealers(risk_threshold=0.7)
print(f"High-risk dealers: {len(high_risk)}")
```

### **4. Understanding Model Decisions**

```python
from src.model_explainer import ModelExplainer

explainer = ModelExplainer()

# Plot feature importance
explainer.plot_feature_importance(model, X_train)

# Plot SHAP summary
explainer.plot_summary_plot(model, X_train)

# Individual prediction explanation
explainer.plot_waterfall_plot(model, X_train.iloc[0:1])
```

## 🔧 Configuration

### **Model Configuration**

Edit `src/model_config.py` to customize:

```python
# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Hyperparameter tuning ranges
HYPERPARAMETER_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}
```

### **File Paths**

```python
# Output paths
MODEL_OUTPUT_DIR = "models"
PREDICTIONS_OUTPUT_DIR = "predictions"
PLOTS_OUTPUT_DIR = "plots"
```

## 📁 Output Files

The training pipeline creates several output files:

```
models/
├── dealer_churn_model.pkl          # Trained model
├── feature_importance.csv          # Feature importance scores
└── model_metadata.json            # Model metadata

predictions/
├── dealer_predictions.csv         # All dealer predictions
├── high_risk_dealers.csv          # High-risk dealers
└── prediction_summary.json        # Prediction statistics

plots/
├── confusion_matrix.png           # Confusion matrix
├── roc_curve.png                  # ROC curve
├── feature_importance.png         # Feature importance plot
└── shap_summary.png               # SHAP summary plot
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Missing Data File**
   ```
   Error: offset_features.csv not found
   Solution: Run feature engineering pipeline first
   ```

2. **Memory Issues**
   ```
   Error: Out of memory during training
   Solution: Reduce data size or use quick training
   ```

3. **Model Loading Errors**
   ```
   Error: Model file not found
   Solution: Train model first or check file paths
   ```

### **Performance Tips**

1. **For Large Datasets**: Use `run_quick_training()` for faster results
2. **For Production**: Use `run_complete_training()` with hyperparameter tuning
3. **For Testing**: Use smaller data samples or limit features

## 📈 Monitoring and Maintenance

### **Regular Tasks**

1. **Retrain Model**: Monthly or when data patterns change
2. **Monitor Performance**: Track accuracy and business metrics
3. **Update Features**: Add new features as business evolves
4. **Review High-Risk Dealers**: Take action on predicted churners

### **Model Validation**

```python
# Check model performance
performance = pipeline.get_model_performance_summary()

# Validate on new data
new_predictions = pipeline.run_production_predictions("new_data.csv")

# Compare with business outcomes
# (implement your own validation logic)
```

## 🎯 Next Steps

1. **Run the example**: `python example_model_training.py`
2. **Customize configuration**: Edit `src/model_config.py`
3. **Train your model**: Use the appropriate training method
4. **Deploy predictions**: Use production prediction functions
5. **Monitor results**: Track model performance and business impact

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your data format matches expected structure
3. Ensure all dependencies are installed
4. Check file paths and permissions
5. Review error messages for specific guidance
