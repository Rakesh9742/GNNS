import torch
from torch_geometric.utils import from_networkx
import pickle
import pandas as pd

# Step 4: Load the graph and dataset
graph_path = r"D:\GNNs\input data\graph_data.pkl"
dataset_path = r"D:\GNNs\input data\preprocessed_dataset.csv"

# Load the graph
with open(graph_path, "rb") as f:
    G = pickle.load(f)
print("Graph loaded successfully!")

# Load the dataset
data = pd.read_csv(dataset_path)
features = torch.tensor(data.drop(['label'], axis=1).values, dtype=torch.float)  # Node features
labels = torch.tensor(data['label'].values, dtype=torch.long)  # Node labels

# Convert the graph to PyTorch Geometric Data
data = from_networkx(G)
data.x = features  # Assign node features
data.y = labels    # Assign node labels

# Save the PyTorch Geometric Data object
torch_data_path = r"D:\GNNs\input data\graph_data.pt"
torch.save(data, torch_data_path)
print(f"PyTorch Geometric Data saved to: {torch_data_path}")
