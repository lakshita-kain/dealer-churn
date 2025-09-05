"""
Feature engineer module for dealer churn analysis.
Handles creation of various features from processed data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from .config import CLUB_HIERARCHY, SETTLED_CLAIM_STATUSES


class FeatureEngineer:
    """Handles feature engineering for dealer churn analysis."""
    
    def __init__(self, sales_journey, customer_master, reference_date):
        """Initialize with required data."""
        self.sales_journey = sales_journey.copy()
        self.customer_master = customer_master.copy()
        self.reference_date = reference_date
        
    def add_dealership_age(self):
        """Add dealership age feature to customer master."""
        self.customer_master['dealership_age'] = (
            datetime.now() - pd.to_datetime(self.customer_master['creation_date'])
        ).dt.days // 365
        return self.customer_master
    
    def create_monthly_sales_features(self):
        """Create monthly sales features and trends."""
        # Calculate total sales trajectory over last 1 year
        monthly_sales = self.sales_journey.groupby(['dealer_code', 'period']).agg(
            total_sales=('ndp_value', 'sum'),
            total_invoices=('invoice_number', 'nunique'),
            total_units=('volume', 'sum'),
        ).reset_index()

        # Calculate average metrics
        period_count = monthly_sales['period'].nunique()
        monthly_sales['avg_invoices'] = monthly_sales['total_invoices'] / period_count
        monthly_sales['avg_sales'] = monthly_sales['total_sales'] / period_count
        monthly_sales['avg_units_purchased'] = monthly_sales['total_units'] / period_count

        # Add territory and dealer club information
        monthly_sales = pd.merge(
            monthly_sales, 
            self.customer_master[['dealer_code', 'territory_code', 'dealer_club_category']], 
            on='dealer_code', 
            how='left'
        )

        return monthly_sales
    
    def create_territory_and_club_features(self, monthly_sales):
        """Create territory and dealer club level features."""
        # Territory level sales trends
        terrwise_monthly_sales = monthly_sales.groupby(['territory_code', 'period'])[
            ['total_sales', 'total_invoices', 'total_units']
        ].mean().reset_index().rename(columns={
            'total_sales': 'avg_sales_of_dealers_in_same_territory',
            'total_invoices': 'avg_orders_of_dealers_in_same_territory',
            'total_units': 'avg_units_of_dealers_in_same_territory'
        })

        # Dealer club level sales trends
        dealerclubwise_monthly_sales = monthly_sales.groupby(['dealer_club_category', 'period'])[
            ['total_sales', 'total_invoices', 'total_units']
        ].mean().reset_index().rename(columns={
            'total_sales': 'avg_sales_of_dealers_in_same_dealer_club_category',
            'total_invoices': 'avg_orders_of_dealers_in_dealer_club_category',
            'total_units': 'avg_units_of_dealers_in_dealer_club_category'
        })

        return terrwise_monthly_sales, dealerclubwise_monthly_sales
    
    def process_sas_data(self, sas_monthly_data):
        """Process SAS monthly data and add features."""
        sas_monthly_data = sas_monthly_data.copy()
        sas_monthly_data['dealer_code'] = sas_monthly_data['dealer_code'].astype(str)
        
        # Update period for SAS data
        sas_monthly_data['upd_period'] = (
            pd.to_datetime(sas_monthly_data['period'], format='%Y-%m') - pd.DateOffset(months=1)
        ).dt.strftime('%Y-%m')
        
        # Filter for reference date
        sas_monthly_data = sas_monthly_data[
            sas_monthly_data['upd_period'] <= self.reference_date.strftime('%Y-%m')
        ]
        
        sas_monthly_data['period'] = sas_monthly_data['upd_period'].astype(str)
        
        return sas_monthly_data
    
    def create_rotation_features(self, monthly_data, sas_monthly_data):
        """Create rotation features and territory/club level rotations."""
        # Merge SAS data
        monthly_data = pd.merge(
            monthly_data, 
            sas_monthly_data[['dealer_code', 'sas_amount', 'period']], 
            on=['dealer_code', 'period'], 
            how='left'
        )
        
        # Calculate rotation
        monthly_data['rotation'] = monthly_data['total_sales'] / monthly_data['sas_amount']
        
        # Territory and dealer club level rotations
        terrwise_sas = monthly_data.groupby(['territory_code', 'period'])['rotation'].mean().reset_index(
            name='terrwise_rotation'
        )
        dealerclub_wise_sas = monthly_data.groupby(['dealer_club_category', 'period'])['rotation'].mean().reset_index(
            name='dealerclub_wise_rotation'
        )
        
        return monthly_data, terrwise_sas, dealerclub_wise_sas
    
    def create_club_movement_features(self, sas_monthly_data):
        """Create dealer club movement features."""
        sas_monthly_data = sas_monthly_data.copy()
        
        # Get previous dealer club
        sas_monthly_data['prev_dealer_club'] = sas_monthly_data.groupby('dealer_code')['dealer_club'].shift(1)
        
        # Map club levels
        sas_monthly_data['curr_level'] = sas_monthly_data['dealer_club'].map(CLUB_HIERARCHY)
        sas_monthly_data['prev_level'] = sas_monthly_data['prev_dealer_club'].map(CLUB_HIERARCHY)
        
        # Classify movement
        def classify_movement(row):
            if pd.isna(row['prev_level']):
                return np.nan
            elif row['curr_level'] > row['prev_level']:
                return 'Promoted'
            elif row['curr_level'] < row['prev_level']:
                return 'Demoted'
            else:
                return 'No Change'
        
        sas_monthly_data['club_movement'] = sas_monthly_data.apply(classify_movement, axis=1)
        
        # Count movements
        movement_counts = (
            sas_monthly_data
            .groupby(['dealer_code', 'club_movement'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        
        return movement_counts
    
    def process_outstanding_data(self, outstanding_df):
        """Process outstanding data and format period."""
        outstanding_df = outstanding_df.copy()
        outstanding_df['period'] = pd.to_datetime(outstanding_df['period'], format='%Y%m').dt.strftime('%Y-%m')
        return outstanding_df
    
    def create_credit_note_features(self, credit_note_df):
        """Create credit note features."""
        credit_note_df = credit_note_df.copy()
        
        monthly_credit_note = credit_note_df.groupby(['dealer_code', 'period']).agg(
            total_credit_note_value=('note_value', 'sum'),
            avg_credit_note_value=('note_value', 'mean')        
        ).reset_index()
        
        return monthly_credit_note
    
    def create_order_type_features(self, sales_data, orders_df):
        """Create online vs offline order features."""
        # Clean invoice numbers
        sales_data['invoice_number'] = sales_data['invoice_number'].apply(
            lambda x: str(x).split('.')[0] if pd.notna(x) else x
        )
        
        # Rename columns for merging
        orders_df = orders_df.rename(columns={
            'products_invoiceNumber': 'invoice_number', 
            'Dealercode': 'dealer_code'
        })
        
        # Merge to identify order types
        merged_orders = pd.merge(
            sales_data[['dealer_code', 'invoice_number', 'invoice_date', 'volume', 'indicator_cancel', 'ndp_value', 'invoice_type', 'period']], 
            orders_df[['dealer_code', 'invoice_number']],
            on=['dealer_code', 'invoice_number'],
            how='left',
            indicator=True
        )
        
        # Classify order types
        merged_orders['order_type'] = np.where(merged_orders['_merge'] == 'both', 'online', 'offline')
        
        # Aggregate by dealer and period
        order_types = merged_orders.drop_duplicates(subset='invoice_number').groupby(['dealer_code', 'period'])['order_type'].value_counts().unstack(fill_value=0).reset_index()
        order_types['total_orders'] = order_types[['online', 'offline']].sum(axis=1)
        
        return order_types
    
    def create_claims_features(self, claims_data):
        """Create claims-related features."""
        claims_data = claims_data.copy()
        
        # Convert to period format
        claims_data['period'] = pd.to_datetime(claims_data['ClaimDate']).dt.to_period('M').astype(str)
        
        # Create claim state
        claims_data['claim_state'] = claims_data['ClaimStatus'].apply(
            lambda x: 'Settled' if x in SETTLED_CLAIM_STATUSES else 'Active'
        )
        
        # Count claims by type
        total_claims = (
            claims_data
            .groupby(['DealerCode', 'period'])['ClaimNo']
            .nunique()
            .reset_index(name='unique_claims')
        )
        
        settled_claims = (
            claims_data[claims_data['claim_state'] == 'Settled']
            .drop_duplicates("ClaimNo")
            .groupby(['DealerCode', 'period'])['ClaimNo']
            .count()
            .reset_index(name='settled_claims')
        )
        
        # Merge and calculate active claims
        claim_count = pd.merge(total_claims, settled_claims, on=['DealerCode', 'period'], how='left')
        claim_count["active_claims"] = claim_count['unique_claims'] - claim_count['settled_claims']
        
        # Rename columns
        claim_count = claim_count.rename(columns={'DealerCode': 'dealer_code'})
        
        return claim_count
    
    def create_visit_features(self, visits_data):
        """Create visit-related features."""
        visits_data = visits_data.copy()
        
        # Filter completed visits
        visits_data = visits_data[visits_data['completed'] == True]
        visits_data['period'] = pd.to_datetime(visits_data['updatedAt']).dt.to_period('M').astype(str)
        
        # Count visits by dealer and period
        visit_count = visits_data.groupby(['dealerCode', 'period'])['_id'].nunique().reset_index(name='visit_count')
        visit_count = visit_count.rename(columns={'dealerCode': 'dealer_code'})
        
        return visit_count
    
    def create_cm_labels(self, monthly_data):
        """Create current month labels for time-based features."""
        monthly_data = monthly_data.copy()
        
        # Ensure period is datetime for correct sorting
        if monthly_data['period'].dtype == 'object':
            monthly_data['period'] = pd.to_datetime(monthly_data['period'], format='%Y-%m')
        monthly_data = monthly_data.sort_values(['dealer_code', 'period'])
        
        def assign_cm_labels(group):
            max_period = group['period'].max()
            min_period = group['period'].min()
            
            # Create a full month range
            full_range = pd.date_range(start=min_period, end=max_period, freq='MS')
            
            # Map actual months to labels
            label_map = {}
            for i, month in enumerate(sorted(full_range, reverse=True), start=1):
                label_map[month] = f'cm-{i}'
            
            # Assign labels only if period exists in label_map
            group['cm_label'] = group['period'].map(label_map)
            return group
        
        monthly_data = monthly_data.groupby('dealer_code', group_keys=False).apply(assign_cm_labels)
        
        # Remove cm-13 (13th month)
        monthly_data = monthly_data[monthly_data['cm_label'] != "cm-13"]
        
        return monthly_data
