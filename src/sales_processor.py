"""
Sales processor module for dealer churn analysis.
Handles sales data processing and feature engineering.
"""

import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from .config import (
    REFERENCE_DATE_OFFSET_DAYS, 
    ANALYSIS_PERIOD_YEARS, 
    CHURN_THRESHOLD_DAYS,
    MIN_STREAK_FOR_CHURN,
    MIN_THRESHOLD_DATE
)


class SalesProcessor:
    """Handles sales data processing and feature engineering."""
    
    def __init__(self, sales_data, customer_master):
        """Initialize with sales data and customer master."""
        self.sales_data = sales_data.copy()
        self.customer_master = customer_master.copy()
        self.reference_date = None
        self.one_year_ago = None
        
    def prepare_sales_data(self):
        """Prepare sales data by filtering and setting reference dates."""
        # Convert invoice_date to datetime
        self.sales_data['invoice_date'] = pd.to_datetime(self.sales_data['invoice_date'])
        
        # Set reference date (last day of previous month)
        max_date = self.sales_data['invoice_date'].max()
        self.reference_date = max_date.replace(day=1) - pd.Timedelta(days=REFERENCE_DATE_OFFSET_DAYS)
        
        # Set one year ago date
        self.one_year_ago = self.reference_date - pd.DateOffset(years=ANALYSIS_PERIOD_YEARS)
        
        # Filter data to exclude current month and limit to analysis period
        self.sales_data = self.sales_data[
            (self.sales_data['invoice_date'] <= self.reference_date)
        ]
        
        return self.sales_data
    
    def calculate_churn_labels(self):
        """Calculate churn status based on last billing date."""
        # Get last invoice date and calculate days since last billing
        last_billed = self.sales_data.groupby('dealer_code')['invoice_date'].max().reset_index(
            name='last_invoice_date'
        )
        last_billed['last_billed_days'] = (
            self.reference_date - last_billed['last_invoice_date']
        ).dt.days
        
        # Assign churn status
        last_billed['churn_status'] = np.where(
            last_billed['last_billed_days'] > CHURN_THRESHOLD_DAYS, 
            'Churned', 
            'Active'
        )
        
        return last_billed
    
    def calculate_sales_streaks(self):
        """Calculate maximum no-sales streaks for each dealer."""
        # Prepare period data
        self.sales_data['period'] = self.sales_data['period'].astype(str)
        all_periods = self.sales_data['period'].unique()
        all_dealers = self.sales_data['dealer_code'].unique()
        
        # Create full index for all dealer-period combinations
        full_index = pd.DataFrame(
            product(all_dealers, all_periods), 
            columns=['dealer_code', 'period']
        )
        
        # Get actual sales data
        monthly_sales = self.sales_data.groupby(['dealer_code', 'period'])['ndp_value'].sum().reset_index()
        
        # Merge to get complete picture
        full_sales = pd.merge(full_index, monthly_sales, on=['dealer_code', 'period'], how='left')
        full_sales['no_sales_flag'] = full_sales['ndp_value'].isna().astype(int)
        full_sales['ndp_value'] = full_sales['ndp_value'].fillna(0)
        
        # Calculate streaks
        streak_info_df = full_sales.groupby('dealer_code').apply(self._get_max_streak_info).reset_index()
        
        return streak_info_df
    
    def _get_max_streak_info(self, group):
        """Helper function to calculate maximum streak information for a group."""
        max_streak = 0
        current_streak = 0
        streak_start = None
        max_streak_start = None

        for idx, row in group.iterrows():
            if row['no_sales_flag'] == 1:
                if current_streak == 0:
                    streak_start = row['period']
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_start = streak_start
            else:
                current_streak = 0
                streak_start = None

        return pd.Series({
            'max_no_sales_streak': max_streak,
            'max_streak_start_period': max_streak_start
        })
    
    def calculate_threshold_dates(self, last_billed, streak_info_df):
        """Calculate threshold dates for analysis period."""
        # Merge streak information
        last_billed = pd.merge(last_billed, streak_info_df, on="dealer_code", how='left')
        
        # Convert period to datetime
        last_billed['max_streak_start_period'] = pd.to_datetime(last_billed['max_streak_start_period'])
        
        # Apply conditions for Active and Churned dealers
        condition1 = (
            (last_billed['churn_status'] == "Churned") &
            (last_billed['max_no_sales_streak'] > MIN_STREAK_FOR_CHURN) &
            (last_billed['max_streak_start_period'] > pd.to_datetime('2022-11'))
        )

        condition2 = (
            (last_billed['churn_status'] == "Active") &
            (
                (last_billed['max_streak_start_period'] <= self.one_year_ago) |
                (last_billed['max_streak_start_period'].isna())
            )
        )

        # Apply conditions
        last_billed['threshold_end_date'] = np.where(
            condition1,
            last_billed['max_streak_start_period'],
            pd.NaT  
        )

        # Apply condition2 only where condition1 is False and condition2 is True
        mask = (~condition1) & (condition2)
        last_billed.loc[mask, 'threshold_end_date'] = self.reference_date

        # Fill remaining nulls
        last_billed['threshold_end_date'] = last_billed['threshold_end_date'].fillna(
            last_billed['max_streak_start_period']
        )
        
        last_billed['threshold_end_date'] = pd.to_datetime(last_billed['threshold_end_date'])

        # Remove dealers without complete 1 year journey
        last_billed = last_billed[last_billed['threshold_end_date'] > pd.to_datetime(MIN_THRESHOLD_DATE)]

        # Calculate start date
        last_billed['threshold_start_date'] = last_billed['threshold_end_date'] - pd.DateOffset(months=12)
        
        return last_billed
    
    def create_sales_journey(self, last_billed):
        """Create sales journey data within threshold dates."""
        # Convert dates to string format for merging
        last_billed['threshold_end_date'] = (last_billed['threshold_end_date'].dt.to_period('M')).astype(str)
        last_billed['threshold_start_date'] = (last_billed['threshold_start_date'].dt.to_period('M')).astype(str)
        
        # Merge with sales data
        sales_journey = pd.merge(self.sales_data, last_billed, on='dealer_code', how='left')
        
        # Filter by threshold dates
        sales_journey['invoice_date'] = pd.to_datetime(sales_journey['invoice_date'], errors='coerce')
        sales_journey = sales_journey[
            (sales_journey['invoice_date'] >= sales_journey['threshold_start_date']) &
            (sales_journey['invoice_date'] <= sales_journey['threshold_end_date'])
        ].copy()
        
        return sales_journey
    
    def calculate_days_between_purchases(self, sales_journey):
        """Calculate average days between purchases for each dealer."""
        sales_journey['invoice_date'] = pd.to_datetime(sales_journey['invoice_date'])
        sales_journey = sales_journey.sort_values(['dealer_code', 'invoice_date'])
        
        # Calculate days between consecutive purchases
        sales_journey['days_between_purchases'] = sales_journey.groupby(['dealer_code'])['invoice_date'].diff().dt.days
        
        # Calculate average days between purchases per dealer
        days_between_purchase = sales_journey.groupby('dealer_code')['days_between_purchases'].mean().reset_index(
            name="avg_days_between_purchase"
        )
        
        return days_between_purchase
    
    def process_sales_data(self):
        """Main method to process all sales data."""
        print("Preparing sales data...")
        self.prepare_sales_data()
        
        print("Calculating churn labels...")
        last_billed = self.calculate_churn_labels()
        
        print("Calculating sales streaks...")
        streak_info_df = self.calculate_sales_streaks()
        
        print("Calculating threshold dates...")
        last_billed = self.calculate_threshold_dates(last_billed, streak_info_df)
        
        print("Creating sales journey...")
        sales_journey = self.create_sales_journey(last_billed)
        
        print("Calculating days between purchases...")
        days_between_purchase = self.calculate_days_between_purchases(sales_journey)
        
        return {
            'last_billed': last_billed,
            'sales_journey': sales_journey,
            'days_between_purchase': days_between_purchase
        }
