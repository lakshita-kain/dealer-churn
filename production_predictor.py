#!/usr/bin/env python3
"""
Production predictor script for dealer churn prediction.
This script can be used in production to predict churn for all dealers.
"""

import sys
import os
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_pipeline import ModelPipeline


def predict_all_dealers(data_path=None, output_path=None, risk_threshold=0.7):
    """Predict churn for all dealers in the dataset."""
    print("🚀 Starting Production Prediction for All Dealers")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize pipeline
        pipeline = ModelPipeline()
        
        # Run production predictions
        predictions = pipeline.run_production_prediction_pipeline(
            data_path=data_path,
            save_results=True
        )
        
        # Get high-risk dealers
        high_risk = pipeline.get_high_risk_dealers(risk_threshold=risk_threshold)
        
        # Print summary
        print(f"\n📊 Prediction Summary:")
        print(f"Total dealers analyzed: {len(predictions)}")
        print(f"High-risk dealers (≥{risk_threshold}): {len(high_risk) if high_risk is not None else 0}")
        
        if high_risk is not None and len(high_risk) > 0:
            print(f"\n🚨 Top 10 High-Risk Dealers:")
            top_10 = high_risk.head(10)
            for idx, row in top_10.iterrows():
                print(f"  {row['dealer_code']}: {row['churn_probability']:.4f} ({row['risk_category']})")
        
        # Save high-risk dealers separately if requested
        if output_path and high_risk is not None:
            high_risk_file = output_path.replace('.csv', '_high_risk.csv')
            high_risk.to_csv(high_risk_file, index=False)
            print(f"\n💾 High-risk dealers saved to: {high_risk_file}")
        
        print(f"\n✅ Production prediction completed successfully!")
        return predictions
        
    except Exception as e:
        print(f"❌ Error in production prediction: {str(e)}")
        raise


def predict_specific_dealers(dealer_codes, data_path=None, output_path=None):
    """Predict churn for specific dealers."""
    print(f"🎯 Starting Prediction for {len(dealer_codes)} Specific Dealers")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize pipeline
        pipeline = ModelPipeline()
        
        # Run specific dealer predictions
        predictions = pipeline.run_specific_dealer_prediction(
            dealer_codes=dealer_codes,
            data_path=data_path,
            save_results=True
        )
        
        if predictions is not None:
            print(f"\n📊 Prediction Results:")
            for idx, row in predictions.iterrows():
                print(f"  Dealer {row['dealer_code']}: {row['prediction_label']} "
                      f"(Probability: {row['churn_probability']:.4f}, Risk: {row['risk_category']})")
        
        print(f"\n✅ Specific dealer prediction completed successfully!")
        return predictions
        
    except Exception as e:
        print(f"❌ Error in specific dealer prediction: {str(e)}")
        raise


def explain_dealer_prediction(dealer_code, data_path=None):
    """Explain prediction for a specific dealer."""
    print(f"🔍 Explaining Prediction for Dealer {dealer_code}")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = ModelPipeline()
        
        # Get dealer explanation
        explanation = pipeline.explain_dealer_prediction(dealer_code)
        
        if explanation is not None:
            print(f"\n✅ Explanation generated for dealer {dealer_code}")
        else:
            print(f"⚠️  No explanation available for dealer {dealer_code}")
        
        return explanation
        
    except Exception as e:
        print(f"❌ Error in dealer explanation: {str(e)}")
        raise


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Dealer Churn Production Predictor')
    parser.add_argument('--mode', choices=['all', 'specific', 'explain'], 
                       default='all', help='Prediction mode')
    parser.add_argument('--data', default='offset_features.csv', 
                       help='Path to input data file')
    parser.add_argument('--output', help='Path to output file')
    parser.add_argument('--dealers', nargs='+', 
                       help='List of dealer codes for specific prediction')
    parser.add_argument('--dealer', help='Single dealer code for explanation')
    parser.add_argument('--risk-threshold', type=float, default=0.7,
                       help='Risk threshold for high-risk classification')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        print("💡 Make sure the feature engineering pipeline has been run first.")
        return 1
    
    try:
        if args.mode == 'all':
            # Predict for all dealers
            predictions = predict_all_dealers(
                data_path=args.data,
                output_path=args.output,
                risk_threshold=args.risk_threshold
            )
            
        elif args.mode == 'specific':
            # Predict for specific dealers
            if not args.dealers:
                print("❌ Error: --dealers argument required for specific prediction mode")
                return 1
            
            predictions = predict_specific_dealers(
                dealer_codes=args.dealers,
                data_path=args.data,
                output_path=args.output
            )
            
        elif args.mode == 'explain':
            # Explain specific dealer prediction
            if not args.dealer:
                print("❌ Error: --dealer argument required for explanation mode")
                return 1
            
            explanation = explain_dealer_prediction(
                dealer_code=args.dealer,
                data_path=args.data
            )
        
        print(f"\n🎉 Production prediction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error in production prediction: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
