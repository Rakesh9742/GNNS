import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# Step 5: Load the PyTorch Geometric Data
torch_data_path = r"D:\GNNs\input data\graph_data.pt"
data = torch.load(torch_data_path)
print(f"PyTorch Geometric Data loaded! Number of nodes: {data.num_nodes}, Number of edges: {data.num_edges}")

# Step 5.1: Define the GAT Model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 5.2: Initialize the Model, Optimizer, and Loss Function
model = GAT(
    in_channels=data.x.size(1),  # Feature dimension
    hidden_channels=64,          # Number of hidden units
    out_channels=2,              # Binary classification
    heads=8                      # Number of attention heads
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Step 5.3: Train the Model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training Loop
print("Starting training...")
for epoch in range(1, 201):  # Train for 200 epochs
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

# Step 5.4: Save the Trained Model
model_path = r"D:\GNNs\input data\gat_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model trained and saved to: {model_path}")
