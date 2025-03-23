from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from sklearn.svm import OneClassSVM
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load stored model or initialize new one
try:
    with open("keystroke_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = OneClassSVM(kernel='rbf', gamma='auto')
    trained = False
else:
    trained = True

@app.route("/train", methods=["POST"])
def train_model():
    global model, trained
    data = request.json.get("keystrokes", [])
    if len(data) < 5:
        return jsonify({"error": "Need more samples to train"}), 400
    
    X = np.array(data)
    model.fit(X)
    trained = True
    
    with open("keystroke_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return jsonify({"message": "Model trained successfully"})

@app.route("/verify", methods=["POST"])
def verify_user():
    if not trained:
        return jsonify({"error": "Model not trained"}), 400
    
    sample = request.json.get("keystroke", [])
    X_test = np.array(sample).reshape(1, -1)
    prediction = model.predict(X_test)
    
    if prediction[0] == 1:
        return jsonify({"result": "Authenticated"})
    else:
        return jsonify({"result": "Verification failed, use secondary MFA"})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
