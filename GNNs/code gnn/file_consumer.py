import socket
import json
import torch
from flask_socketio import SocketIO
from torch_geometric.nn import GATConv
from app import send_prediction_to_frontend

# Load the graph model and data
model_path = r"D:\GNNs\input data\gat_model.pt"
graph_data_path = r"D:\GNNs\input data\graph_data.pt"
data = torch.load(graph_data_path)  # Load graph structure

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

# Initialize the GAT model
gat_model = GAT(in_channels=data.x.size(1), hidden_channels=64, out_channels=2, heads=8)
gat_model.load_state_dict(torch.load(model_path))
gat_model.eval()

# Consumer setup
HOST = "127.0.0.1"
PORT = 65432
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print("Consumer: Connected to producer, waiting for data...")

try:
    while True:
        # Receive data from the producer
        data_bytes = client_socket.recv(1024)
        if not data_bytes:
            break

        # Decode and parse the data
        try:
            data_point = json.loads(data_bytes.decode().strip())
        except json.JSONDecodeError as e:
            print(f"Consumer Error: {e}")
            continue

        # Prepare feature tensor
        feature_vector = torch.tensor([[data_point[key] for key in data_point if key != "label"]], dtype=torch.float)

        # Predict using the model
        with torch.no_grad():
            data.x[0] = feature_vector[0]
            out = gat_model(data.x, data.edge_index)
            prediction = out[0].argmax(dim=0).item()

        # Determine label
        label = "Anomaly" if prediction == 1 else "Normal"
        print(f"Consumer: Received -> ID: {data_point['id']}, Prediction: {label}")

        # Send prediction to the frontend
        send_prediction_to_frontend({"id": data_point["id"], "prediction": label})

except KeyboardInterrupt:
    print("Consumer: Stopped.")
finally:
    client_socket.close()
