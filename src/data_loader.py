"""
Data loader module for dealer churn analysis.
Handles loading data from various DLT tables.
"""

import pandas as pd
from pyspark.sql.functions import col
from dlt_utils import DLTReader
from .config import DLT_CONFIG, DEALER_GROUP


class DataLoader:
    """Handles loading data from various DLT tables."""
    
    def __init__(self):
        """Initialize the data loader with DLT configurations."""
        self.dlt_reader = DLTReader(
            catalog=DLT_CONFIG["catalog"],
            schema=DLT_CONFIG["schema"],
        )
        self.sap_reader = DLTReader(
            catalog=DLT_CONFIG["catalog"],
            schema=DLT_CONFIG["sap_schema"],
        )
        self.jkc_reader = DLTReader(
            catalog=DLT_CONFIG["catalog"],
            schema=DLT_CONFIG["jkc_schema"],
        )
        self.dealer_list = None
        
    def load_customer_master(self):
        """Load customer master data and filter for specific dealer group."""
        customer_data = self.dlt_reader.read_table("customer_master")
        customer_data = customer_data.filter(col("dealer_group") == DEALER_GROUP)
        customer_master = customer_data.toPandas()
        
        # Store dealer list for other data loading operations
        self.dealer_list = customer_master['dealer_code'].to_list()
        
        return customer_master
    
    def load_sales_data(self):
        """Load sales data filtered for specific dealers and conditions."""
        if not self.dealer_list:
            raise ValueError("Customer master must be loaded first to get dealer list")
            
        sales = self.dlt_reader.read_table("sales_data")
        sales = sales.filter(
            (col("dealer_code").isin([str(dealer) for dealer in self.dealer_list])) &
            (col("indicator_cancel").isNull()) &
            (~col("invoice_type").isin(['S1', 'S2', 'S3', 'S4']))
        )
        
        sales_data = sales.toPandas()
        sales_data['dealer_code'] = sales_data['dealer_code'].astype(str)
        
        return sales_data
    
    def load_claims_data(self):
        """Load claims data for specific dealers."""
        if not self.dealer_list:
            raise ValueError("Customer master must be loaded first to get dealer list")
            
        claims = self.dlt_reader.read_table("claims_data")
        claims = claims.filter(claims.DealerCode.isin([str(dealer) for dealer in self.dealer_list]))
        return claims.toPandas()
    
    def load_territory_master(self):
        """Load territory master data."""
        territory_master = self.dlt_reader.read_table("territory_master")
        return territory_master.toPandas()
    
    def load_visits_data(self):
        """Load visits data for specific dealers."""
        if not self.dealer_list:
            raise ValueError("Customer master must be loaded first to get dealer list")
            
        visits = self.dlt_reader.read_table("visits_flat")
        visits = visits.filter(visits.dealerCode.isin([str(dealer) for dealer in self.dealer_list]))
        return visits.toPandas()
    
    def load_sas_monthly_data(self):
        """Load SAS monthly club mapping data."""
        if not self.dealer_list:
            raise ValueError("Customer master must be loaded first to get dealer list")
            
        sas_monthly = self.dlt_reader.read_table("monthly_club_mapping")
        sas_monthly = sas_monthly.filter(sas_monthly.dealer_code.isin([str(dealer) for dealer in self.dealer_list]))
        return sas_monthly.toPandas()
    
    def load_credit_note_data(self):
        """Load credit note data for specific dealers."""
        if not self.dealer_list:
            raise ValueError("Customer master must be loaded first to get dealer list")
            
        credit_note = self.dlt_reader.read_table("credit_note")
        credit_note = credit_note.filter(credit_note.dealer_code.isin([str(dealer) for dealer in self.dealer_list]))
        return credit_note.toPandas()
    
    def load_product_master(self):
        """Load product master data."""
        pm = self.dlt_reader.read_table("product_master")
        return pm.toPandas()
    
    def load_outstanding_data(self):
        """Load customer financial data from SAP schema."""
        if not self.dealer_list:
            raise ValueError("Customer master must be loaded first to get dealer list")
            
        # Switch to SAP schema
        sap_reader = self.sap_reader
        
        outstanding = sap_reader.read_table("customer_financial")
        outstanding = outstanding.filter(outstanding.dealer_code.isin([str(dealer) for dealer in self.dealer_list]))
        return outstanding.toPandas()
    
    def load_orders_data(self):
        """Load dealer orders data from JKC schema."""
        if not self.dealer_list:
            raise ValueError("Customer master must be loaded first to get dealer list")
            
        # Switch to JKC schema
        jkc_reader = self.jkc_reader
        
        orders = jkc_reader.read_table("orders")
        orders = orders.filter(orders.Dealercode.isin([str(dealer) for dealer in self.dealer_list]))
        orders = orders.select('products_invoiceNumber', 'Dealercode')
        return orders.toPandas()
    
    def load_all_data(self):
        """Load all required data sources."""
        print("Loading customer master...")
        customer_master = self.load_customer_master()
        
        print("Loading sales data...")
        sales_data = self.load_sales_data()
        
        print("Loading claims data...")
        claims_data = self.load_claims_data()
        
        print("Loading territory master...")
        territory_master = self.load_territory_master()
        
        print("Loading visits data...")
        visits_data = self.load_visits_data()
        
        print("Loading SAS monthly data...")
        sas_monthly_data = self.load_sas_monthly_data()
        
        print("Loading credit note data...")
        credit_note_df = self.load_credit_note_data()
        
        print("Loading product master...")
        product_data = self.load_product_master()
        
        print("Loading outstanding data...")
        outstanding_df = self.load_outstanding_data()
        
        print("Loading orders data...")
        orders_df = self.load_orders_data()
        
        return {
            'customer_master': customer_master,
            'sales_data': sales_data,
            'claims_data': claims_data,
            'territory_master': territory_master,
            'visits_data': visits_data,
            'sas_monthly_data': sas_monthly_data,
            'credit_note_df': credit_note_df,
            'product_data': product_data,
            'outstanding_df': outstanding_df,
            'orders_df': orders_df
        }
