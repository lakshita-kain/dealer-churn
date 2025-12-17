"""
Configuration module for dealer churn analysis.
Contains all configurable parameters and constants.
"""

# DLT Configuration
DLT_CONFIG = {
    "catalog": "provisioned-tableau-data",
    "schema": "tableau_delta_tables",
    "sap_schema": "sap_data",
    "jkc_schema": "jkc"
}

# Dealer Configuration
DEALER_GROUP = "Z001"

# Date Configuration
REFERENCE_DATE_OFFSET_DAYS = 1
ANALYSIS_PERIOD_YEARS = 1
CHURN_THRESHOLD_DAYS = 90
MIN_STREAK_FOR_CHURN = 4
MIN_THRESHOLD_DATE = "2022-12"

# Sales Data Configuration
EXCLUDED_INVOICE_TYPES = ['S1', 'S2', 'S3', 'S4']

# Club Hierarchy Configuration
CLUB_HIERARCHY = {
    'Non Starter': 0,
    'Starter': 1,
    'Blue Club': 2,
    'Gold Plus Club': 3,
    'Platinum Club': 4,
    'Diamond Club': 5,
    'Acer Club': 6,
    "Chairman's Club": 7,
    "Chairman's Advisory Club": 8
}

# Claims Configuration
SETTLED_CLAIM_STATUSES = [
    'Inspection Accepted',
    'Inspection Rejected',
    'AI : Inspection Rejected'
]

# File Paths
OUTPUT_FILE_PATH = "offset_features"
