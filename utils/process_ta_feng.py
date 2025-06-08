import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('ta-feng-original/ta_feng_all_months_merged.csv')

# Fill missing values with "Unknown"
data[['AGE_GROUP', 'PIN_CODE']] = data[['AGE_GROUP', 'PIN_CODE']].fillna("Unknown")

# Group by CUSTOMER_ID and take the first non-null value for each group
user_features = (
    data.groupby('CUSTOMER_ID')[['AGE_GROUP', 'PIN_CODE']]
    .first()
    .reset_index()
)

# Fill again in case grouping kept nulls
user_features = user_features.fillna("Unknown")

# Rename columns to match required format
user_features.columns = ['customer_id:token', 'age_group:token', 'pin_code:token']

# Save to file
user_features.to_csv('ta-feng/ta-feng.user', sep='\t', index=False)

# Copy relevant columns
interactions = data[['TRANSACTION_DT', 'CUSTOMER_ID', 'PRODUCT_ID', 'AMOUNT']].copy()

# Convert TRANSACTION_DT to datetime and then to UNIX timestamp
interactions['TRANSACTION_DT'] = pd.to_datetime(interactions['TRANSACTION_DT'], format="%m/%d/%Y", errors='coerce')
interactions['timestamp'] = interactions['TRANSACTION_DT'].astype(np.int64) // 10**9

# Add small unique noise to timestamps to make each unique
interactions['timestamp'] += np.linspace(0, 1, len(interactions), endpoint=False)

# Prepare final dataframe
interactions_df = interactions[['timestamp', 'CUSTOMER_ID', 'PRODUCT_ID', 'AMOUNT']].copy()
interactions_df.columns = ['transaction_date:float', 'customer_id:token', 'product_id:token', 'amount:float']

# Save to file
interactions_path = 'ta-feng/ta-feng.inter'
interactions_df.to_csv(interactions_path, sep='\t', index=False)
