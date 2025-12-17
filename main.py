"""
Main module for dealer churn analysis.
Orchestrates the entire feature engineering pipeline.
"""

from src.data_loader import DataLoader
from src.sales_processor import SalesProcessor
from src.feature_engineer import FeatureEngineer
from src.data_aggregator import DataAggregator


def run_feature_engineering_pipeline():
    """Run the complete feature engineering pipeline."""
    print("🚀 Starting Dealer Churn Feature Engineering Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Load all data
        print("\n📊 Step 1: Loading Data")
        print("-" * 30)
        data_loader = DataLoader()
        data_dict = data_loader.load_all_data()
        
        # Extract data from dictionary
        customer_master = data_dict['customer_master']
        sales_data = data_dict['sales_data']
        claims_data = data_dict['claims_data']
        territory_master = data_dict['territory_master']
        visits_data = data_dict['visits_data']
        sas_monthly_data = data_dict['sas_monthly_data']
        credit_note_df = data_dict['credit_note_df']
        product_data = data_dict['product_data']
        outstanding_df = data_dict['outstanding_df']
        orders_df = data_dict['orders_df']
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Customer Master: {customer_master.shape}")
        print(f"   - Sales Data: {sales_data.shape}")
        print(f"   - Claims Data: {claims_data.shape}")
        print(f"   - Territory Master: {territory_master.shape}")
        print(f"   - Visits Data: {visits_data.shape}")
        print(f"   - SAS Monthly Data: {sas_monthly_data.shape}")
        print(f"   - Credit Note Data: {credit_note_df.shape}")
        print(f"   - Product Data: {product_data.shape}")
        print(f"   - Outstanding Data: {outstanding_df.shape}")
        print(f"   - Orders Data: {orders_df.shape}")
        
        # Step 2: Process sales data
        print("\n🔄 Step 2: Processing Sales Data")
        print("-" * 30)
        sales_processor = SalesProcessor(sales_data, customer_master)
        sales_results = sales_processor.process_sales_data()
        
        last_billed = sales_results['last_billed']
        sales_journey = sales_results['sales_journey']
        days_between_purchase = sales_results['days_between_purchase']
        
        print(f"✅ Sales data processed successfully:")
        print(f"   - Last Billed: {last_billed.shape}")
        print(f"   - Sales Journey: {sales_journey.shape}")
        print(f"   - Days Between Purchase: {days_between_purchase.shape}")
        
        # Step 3: Feature Engineering
        print("\n🔧 Step 3: Feature Engineering")
        print("-" * 30)
        feature_engineer = FeatureEngineer(sales_journey, customer_master, sales_processor.reference_date)
        
        # Add dealership age
        customer_master = feature_engineer.add_dealership_age()
        
        # Create monthly sales features
        monthly_sales = feature_engineer.create_monthly_sales_features()
        
        # Create territory and club features
        terrwise_monthly_sales, dealerclubwise_monthly_sales = feature_engineer.create_territory_and_club_features(monthly_sales)
        
        # Merge territory and club features
        monthly_sales = pd.merge(
            monthly_sales, 
            terrwise_monthly_sales, 
            on=['territory_code', 'period'], 
            how='left'
        ).merge(
            dealerclubwise_monthly_sales, 
            on=['dealer_club_category', 'period'], 
            how='left'
        )
        
        # Process SAS data
        sas_monthly_data = feature_engineer.process_sas_data(sas_monthly_data)
        
        # Create rotation features
        monthly_sales, terrwise_sas, dealerclub_wise_sas = feature_engineer.create_rotation_features(
            monthly_sales, sas_monthly_data
        )
        
        # Merge SAS features
        monthly_sales = pd.merge(
            monthly_sales, 
            terrwise_sas, 
            on=['territory_code', 'period'], 
            how='left'
        ).merge(
            dealerclub_wise_sas, 
            on=['dealer_club_category', 'period'], 
            how='left'
        )
        
        # Create club movement features
        movement_counts = feature_engineer.create_club_movement_features(sas_monthly_data)
        
        # Process outstanding data
        outstanding_df = feature_engineer.process_outstanding_data(outstanding_df)
        
        # Create credit note features
        monthly_credit_note = feature_engineer.create_credit_note_features(credit_note_df)
        
        # Create order type features
        order_types = feature_engineer.create_order_type_features(sales_data, orders_df)
        
        # Create claims features
        claim_count = feature_engineer.create_claims_features(claims_data)
        
        # Create visit features
        visit_count = feature_engineer.create_visit_features(visits_data)
        
        # Create CM labels
        monthly_sales = feature_engineer.create_cm_labels(monthly_sales)
        
        print(f"✅ Features engineered successfully:")
        print(f"   - Monthly Sales Features: {monthly_sales.shape}")
        print(f"   - Movement Counts: {movement_counts.shape}")
        print(f"   - Outstanding Data: {outstanding_df.shape}")
        print(f"   - Credit Note Features: {monthly_credit_note.shape}")
        print(f"   - Order Type Features: {order_types.shape}")
        print(f"   - Claims Features: {claim_count.shape}")
        print(f"   - Visit Features: {visit_count.shape}")
        
        # Step 4: Data Aggregation
        print("\n📈 Step 4: Data Aggregation")
        print("-" * 30)
        data_aggregator = DataAggregator()
        
        final_data = data_aggregator.aggregate_all_data(
            monthly_sales, outstanding_df, monthly_credit_note, order_types, 
            claim_count, visit_count, days_between_purchase, movement_counts, 
            customer_master, territory_master, last_billed
        )
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"Final dataset shape: {final_data.shape}")
        print(f"Output saved to: offset_features.csv")
        
        return final_data
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Import pandas here to avoid circular imports
    import pandas as pd
    
    # Run the pipeline
    final_data = run_feature_engineering_pipeline()
