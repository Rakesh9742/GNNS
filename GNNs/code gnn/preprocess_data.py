import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
# Use the provided path
dataset_path = r"D:\GNNs\input data\UNSW_NB15_training-set.csv"

# Load the dataset
data = pd.read_csv(dataset_path)
print(f"Dataset loaded successfully! Shape: {data.shape}")

# Step 2.1: Handle categorical features
categorical_columns = ['proto', 'service', 'state', 'attack_cat']  # Adjust based on your dataset
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    if col in data.columns:
        data[col] = label_encoders[col].fit_transform(data[col])

# Step 2.2: Normalize numerical features
scaler = StandardScaler()
numerical_columns = [col for col in data.columns if col not in categorical_columns + ['label']]  # Exclude 'label'

data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Save the preprocessed dataset
preprocessed_path = r"D:\GNNs\input data\preprocessed_dataset.csv"
data.to_csv(preprocessed_path, index=False)
print(f"Preprocessed dataset saved to: {preprocessed_path}")
