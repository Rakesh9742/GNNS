from flask import Flask, render_template
from flask_socketio import SocketIO

# Flask app and SocketIO initialization
app = Flask(__name__)
socketio = SocketIO(app)

# Route for the frontend
@app.route("/")
def index():
    return render_template("index.html")

# Emit predictions to the frontend
@app.route("/predictions")
def predictions():
    # Placeholder for testing
    return {"status": "WebSocket is active"}

# Function to emit predictions (used by the consumer)
def send_prediction_to_frontend(data):
    socketio.emit("predictions", data)

if __name__ == "__main__":
    socketio.run(app, debug=True)
