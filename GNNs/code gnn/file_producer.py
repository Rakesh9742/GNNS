import socket
import json
import time

# Path to your dataset
input_file = r"D:\GNNs\input data\preprocessed_dataset.csv"

# Setup server
HOST = '127.0.0.1'
PORT = 65432
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Producer: Waiting for consumer to connect...")

conn, addr = server_socket.accept()
print(f"Producer: Connected to consumer at {addr}")

# Read and stream data
with open(input_file, 'r') as infile:
    header = infile.readline().strip().split(",")  # Read header
    for line in infile:
        # Parse each line into a JSON object
        values = line.strip().split(",")
        data_point = {header[i]: float(values[i]) if i < len(values) - 1 else int(values[i]) for i in range(len(header))}
        
        # Send data
        conn.sendall(json.dumps(data_point).encode() + b'\n')
        print(f"Producer Sent -> ID: {data_point['id']}, Label: {data_point['label']}")
        
        # Simulate real-time delay
        time.sleep(0.5)

conn.close()
server_socket.close()
