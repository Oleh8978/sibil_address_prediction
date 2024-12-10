import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data from the parquet files
data_path = './parquet/'  # Adjust path as needed
train_addresses = pd.read_parquet(data_path + 'train_addresses.parquet')
test_addresses = pd.read_parquet(data_path + 'test_addresses.parquet')
transactions = pd.read_parquet(data_path + 'transactions.parquet')
token_transfers = pd.read_parquet(data_path + 'token_transfers.parquet')
dex_swaps = pd.read_parquet(data_path + 'dex_swaps.parquet')

# Convert columns to numeric where necessary
token_transfers['RAW_AMOUNT_PRECISE'] = pd.to_numeric(token_transfers['RAW_AMOUNT_PRECISE'], errors='coerce')
dex_swaps['AMOUNT_IN'] = pd.to_numeric(dex_swaps['AMOUNT_IN'], errors='coerce')
dex_swaps['AMOUNT_OUT'] = pd.to_numeric(dex_swaps['AMOUNT_OUT'], errors='coerce')

# Feature engineering
# Transactions features
tx_features = transactions.groupby('FROM_ADDRESS').agg({
    'VALUE': ['sum', 'mean', 'count'],
    'TO_ADDRESS': 'nunique'
}).reset_index()
tx_features.columns = ['FROM_ADDRESS', 'TX_SUM', 'TX_MEAN', 'TX_COUNT', 'UNIQUE_TO_ADDRESSES']

# Token transfers features
token_features = token_transfers.groupby('FROM_ADDRESS').agg({
    'RAW_AMOUNT_PRECISE': ['sum', 'mean', 'count'],
    'TO_ADDRESS': 'nunique'
}).reset_index()
token_features.columns = ['FROM_ADDRESS', 'TOKEN_SUM', 'TOKEN_MEAN', 'TOKEN_COUNT', 'UNIQUE_TOKEN_TO']

# DEX swaps features
dex_features = dex_swaps.groupby('ORIGIN_FROM_ADDRESS').agg({
    'AMOUNT_IN': ['sum', 'mean', 'count'],
    'AMOUNT_OUT': 'sum'
}).reset_index()
dex_features.columns = ['ORIGIN_FROM_ADDRESS', 'DEX_IN_SUM', 'DEX_IN_MEAN', 'DEX_COUNT', 'DEX_OUT_SUM']

# Merge features with train_addresses
train_data = train_addresses.merge(tx_features, left_on='ADDRESS', right_on='FROM_ADDRESS', how='left')
train_data = train_data.merge(token_features, left_on='ADDRESS', right_on='FROM_ADDRESS', how='left')
train_data = train_data.merge(dex_features, left_on='ADDRESS', right_on='ORIGIN_FROM_ADDRESS', how='left')

# Fill missing values
train_data.fillna(0, inplace=True)

# Debugging: Check available columns in train_data
print("Columns in train_data:", train_data.columns)

# Prepare training data
columns_to_drop = ['ADDRESS', 'LABEL', 'FROM_ADDRESS_x', 'FROM_ADDRESS_y', 'ORIGIN_FROM_ADDRESS']
columns_to_drop = [col for col in columns_to_drop if col in train_data.columns]

X = train_data.drop(columns=columns_to_drop)
y = train_data['LABEL']

# Ensure no NaN values in target
y = y.dropna()

# Ensure y contains only discrete labels (0, 1)
y = y.astype(int)
print("Unique values in y after cleaning:", y.unique())

# Filter numeric columns for scaling
X_numeric = X.select_dtypes(include=['number'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Proceed with train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Validate model
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Test data preparation
test_data = test_addresses.merge(tx_features, left_on='ADDRESS', right_on='FROM_ADDRESS', how='left')
test_data = test_data.merge(token_features, left_on='ADDRESS', right_on='FROM_ADDRESS', how='left')
test_data = test_data.merge(dex_features, left_on='ADDRESS', right_on='ORIGIN_FROM_ADDRESS', how='left')

# Fill missing values
test_data.fillna(0, inplace=True)

# Drop unnecessary columns dynamically
columns_to_drop_test = ['ADDRESS', 'FROM_ADDRESS_x', 'FROM_ADDRESS_y', 'ORIGIN_FROM_ADDRESS']
columns_to_drop_test = [col for col in columns_to_drop_test if col in test_data.columns]

X_test = test_data.drop(columns=columns_to_drop_test)

# Filter numeric columns for scaling
X_test_numeric = X_test.select_dtypes(include=['number'])
X_test_scaled = scaler.transform(X_test_numeric)

# Predict on test data
test_pred = model.predict(X_test_scaled)

# Save predictions to a CSV
submission = pd.DataFrame({'ADDRESS': test_addresses['ADDRESS'], 'PRED': test_pred})
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
