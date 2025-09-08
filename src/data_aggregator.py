"""
Data aggregator module for dealer churn analysis.
Handles data merging, aggregation, and pivoting operations.
"""

import pandas as pd

from dlt_utils import DLTWriter
from .config import OUTPUT_FILE_PATH
from .utils import standardize_period_column


class DataAggregator:
    """Handles data aggregation and final feature creation."""
    
    def __init__(self):
        """Initialize the data aggregator."""
        pass
    
    def merge_all_features(self, monthly_data, outstanding_df, credit_note_df, 
                          order_types, claim_count, visit_count, days_between_purchase, 
                          movement_counts, customer_master, territory_master, last_billed):
        """Merge all feature data into a single dataset."""
        print("Merging all features...")
        
        # Convert period columns to string for consistent merging
        # monthly_data = standardize_period_column(monthly_data)
        # outstanding_df = standardize_period_column(outstanding_df)
        # credit_note_df = standardize_period_column(credit_note_df)
        # order_types = standardize_period_column(order_types)
        # claim_count = standardize_period_column(claim_count)
        # visit_count = standardize_period_column(visit_count)
        
        outstanding_df['period'] = pd.to_datetime(outstanding_df['period'], format='%Y-%m')
        credit_note_df['period'] = pd.to_datetime(credit_note_df['period'], format='%Y-%m')
        order_types['period'] = pd.to_datetime(order_types['period'], format='%Y-%m')
        claim_count['period'] = pd.to_datetime(claim_count['period'], format='%Y-%m')
        visit_count['period'] = pd.to_datetime(visit_count['period'], format='%Y-%m')
        
        # Merge outstanding data
        monthly_data = pd.merge(monthly_data, outstanding_df, on=['dealer_code', 'period'], how='left')
        
        # Merge credit note data
        monthly_data = pd.merge(monthly_data, credit_note_df, on=['dealer_code', 'period'], how='left')
        
        # Merge order type data
        monthly_data = pd.merge(monthly_data, order_types, on=['dealer_code', 'period'], how='left')
        
        # Merge claims data
        monthly_data = pd.merge(monthly_data, claim_count, on=['dealer_code', 'period'], how='left')
        
        # Merge visit data
        monthly_data = pd.merge(monthly_data, visit_count, on=['dealer_code', 'period'], how='left')
        
        # Merge days between purchases
        monthly_data = pd.merge(monthly_data, days_between_purchase, on='dealer_code', how='left')
        
        # Merge movement counts
        monthly_data = pd.merge(monthly_data, movement_counts, on='dealer_code', how='left')
        
        # Merge customer master features
        monthly_data = pd.merge(
            monthly_data, 
            customer_master[['dealer_code', 'territory_code', 'dealer_club_category', 'dealership_age']], 
            on=['dealer_code', 'territory_code', 'dealer_club_category'], 
            how='left'
        )
        
        # Merge territory master
        monthly_data = pd.merge(
            monthly_data, 
            territory_master[['zone', 'territory_code', 'region_name', 'dealer_count']], 
            on='territory_code', 
            how='left'
        )
        
        # Merge churn status
        monthly_data = pd.merge(
            monthly_data, 
            last_billed[['dealer_code', 'last_billed_days', 'churn_status']], 
            on='dealer_code', 
            how='left'
        )
        
        return monthly_data
    
    def create_pivot_features(self, monthly_data):
        """Create pivot table features for time-based analysis."""
        print("Creating pivot features...")
        
        # Ensure period is datetime for correct sorting
        if monthly_data['period'].dtype == 'object':
            monthly_data['period'] = pd.to_datetime(monthly_data['period'], format='%Y-%m-%d')
        monthly_data = monthly_data.sort_values(['dealer_code', 'period'])
        
        # Create pivot table
        pivot_columns = [
            'total_sales', 'total_invoices', 'total_units',
            'avg_invoices', 'avg_sales', 'avg_units_purchased',
            'avg_sales_of_dealers_in_same_territory',
            'avg_orders_of_dealers_in_same_territory',
            'avg_units_of_dealers_in_same_territory',
            'avg_sales_of_dealers_in_same_dealer_club_category',
            'avg_orders_of_dealers_in_dealer_club_category',
            'avg_units_of_dealers_in_dealer_club_category', 
            'sas_amount', 'rotation', 'terrwise_rotation', 'dealerclub_wise_rotation', 
            'exposure', 'collection', 'outstanding', 'outstanding_0_30', 'outstanding_30_45',
            'outstanding_45_60', 'outstanding_60+', 'total_credit_note_value',
            'avg_credit_note_value', 'offline', 'online', 'total_orders',
            'unique_claims', 'settled_claims', 'active_claims', 'visit_count'
        ]
        
        monthly_data_pivot = monthly_data.pivot(
            index='dealer_code', 
            columns='cm_label', 
            values=pivot_columns
        ).reset_index()
        
        # Clean column names
        monthly_data_pivot.columns = [
            '_'.join([str(c) for c in col if c != '']).strip()  
            if isinstance(col, tuple) else str(col)  
            for col in monthly_data_pivot.columns
        ]
        
        return monthly_data_pivot
    
    def create_final_dataset(self, monthly_data_pivot, days_between_purchase, 
                           movement_counts, customer_master, territory_master, last_billed):
        """Create the final aggregated dataset."""
        print("Creating final dataset...")
        
        # Merge pivot data with additional features
        final_data = pd.merge(
            monthly_data_pivot, 
            customer_master[['dealer_code', 'territory_code', 'dealer_club_category', 'dealership_age']], 
            on='dealer_code', 
            how='left'
        )
        
        # Merge days between purchases
        final_data = pd.merge(final_data, days_between_purchase, on='dealer_code', how='left')
        
        # Merge movement counts
        final_data = pd.merge(final_data, movement_counts, on='dealer_code', how='left')
        
        # Merge territory master
        final_data = pd.merge(
            final_data, 
            territory_master[['zone', 'territory_code', 'region_name', 'dealer_count']], 
            on='territory_code', 
            how='left'
        )
        
        # Merge churn status
        final_data = pd.merge(
            final_data, 
            last_billed[['dealer_code', 'last_billed_days', 'churn_status']], 
            on='dealer_code', 
            how='left'
        )
        
        return final_data
    
    def save_dataset(self, final_data, file_path=None):
        """Save the final dataset to CSV."""
        if file_path is None:
            file_path = OUTPUT_FILE_PATH
        dlt_writer = DLTWriter(catalog="provisioned-tableau-data", schema="data_science")
        print(f"Saving dataset to {file_path}...")
        dlt_writer.write_table(final_data, file_path, mode="overwrite")
        print(f"Dataset saved successfully! Shape: {final_data.shape}")
        
        return final_data
    
    def aggregate_all_data(self, monthly_data, outstanding_df, credit_note_df, 
                          order_types, claim_count, visit_count, days_between_purchase, 
                          movement_counts, customer_master, territory_master, last_billed, save_dataset=False):
        """Main method to aggregate all data and create final dataset."""
        print("Starting data aggregation process...")
        
        # Merge all features
        merged_data = self.merge_all_features(
            monthly_data, outstanding_df, credit_note_df, order_types, 
            claim_count, visit_count, days_between_purchase, movement_counts, 
            customer_master, territory_master, last_billed
        )
        
        # Create pivot features
        pivot_data = self.create_pivot_features(merged_data)
        
        # Create final dataset
        final_data = self.create_final_dataset(
            pivot_data, days_between_purchase, movement_counts, 
            customer_master, territory_master, last_billed
        )
        
        # Save dataset
        if save_dataset:
            final_data = self.save_dataset(final_data)
        
        return final_data
