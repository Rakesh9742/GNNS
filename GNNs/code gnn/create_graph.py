import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import pickle

# Step 3: Load the preprocessed dataset
dataset_path = r"D:\GNNs\input data\preprocessed_dataset.csv"
data = pd.read_csv(dataset_path)
print(f"Preprocessed dataset loaded! Shape: {data.shape}")

# Step 3.1: Define nodes and features
nodes = data.index.tolist()  # Use row indices as node IDs
features = data.drop(['label'], axis=1).values  # Node features (excluding label)
labels = data['label'].values  # Node labels

# Step 3.2: Create edges using K-Nearest Neighbors
print("Creating edges using K-Nearest Neighbors...")
edges = []
knn = NearestNeighbors(n_neighbors=5)  # Adjust 'n_neighbors' if needed
knn.fit(features)
for i, neighbors in enumerate(knn.kneighbors(features, return_distance=False)):
    for neighbor in neighbors:
        edges.append((i, neighbor))

# Step 3.3: Create a NetworkX graph
print("Building the graph...")
G = nx.Graph()
G.add_nodes_from(nodes)  # Add nodes
G.add_edges_from(edges)  # Add edges

# Save the graph as a pickle file
graph_path = r"D:\GNNs\input data\graph_data.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G, f)

print(f"Graph created and saved to: {graph_path}")
