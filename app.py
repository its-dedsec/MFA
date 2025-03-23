from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from sklearn.svm import OneClassSVM
from flask_cors import CORS
import streamlit as st
import requests
import matplotlib.pyplot as plt
import time

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

keystroke_data = []  # Store samples for training

@app.route("/train", methods=["POST"])
def train_model():
    global model, trained
    data = request.json.get("keystrokes", [])
    if len(data) < 5:  # Minimum samples required
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
    app.run(debug=True)

# Streamlit UI with Visualization
def streamlit_ui():
    st.title("Keystroke Dynamics-Based MFA")
    st.write("Enter your keystrokes for authentication")
    
    keystroke_data = st.text_input("Type a sample password")
    
    if st.button("Train Model"):
        response = requests.post("http://127.0.0.1:5000/train", json={"keystrokes": [list(map(ord, keystroke_data))]})
        st.write(response.json())
        
        # Visualization: Show training progress
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        st.success("Training Completed!")
        
    if st.button("Verify User"):
        response = requests.post("http://127.0.0.1:5000/verify", json={"keystroke": list(map(ord, keystroke_data))})
        result = response.json()
        st.write(result)
        
        # Visualization: Plot keystroke data
        fig, ax = plt.subplots()
        ax.plot(range(len(keystroke_data)), list(map(ord, keystroke_data)), marker='o', linestyle='-')
        ax.set_title("Keystroke Data Processing")
        ax.set_xlabel("Keystroke Index")
        ax.set_ylabel("ASCII Value")
        st.pyplot(fig)
        
if __name__ == "__main__":
    streamlit_ui()

# JavaScript Frontend (index.html)
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Keystroke Authentication</title>
    <script>
        let keystrokeData = [];
        document.addEventListener("DOMContentLoaded", function() {
            let inputField = document.getElementById("keystrokeInput");
            inputField.addEventListener("keydown", function(event) {
                let time = new Date().getTime();
                keystrokeData.push(time);
            });
        });
        
        function trainModel() {
            fetch("/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ "keystrokes": [keystrokeData] })
            })
            .then(response => response.json())
            .then(data => alert(data.message));
        }
        
        function verifyUser() {
            fetch("/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ "keystroke": keystrokeData })
            })
            .then(response => response.json())
            .then(data => alert(data.result));
        }
    </script>
</head>
<body>
    <h1>Keystroke Dynamics-Based Authentication</h1>
    <input type="text" id="keystrokeInput" placeholder="Type here...">
    <button onclick="trainModel()">Train</button>
    <button onclick="verifyUser()">Verify</button>
</body>
</html>
"""

# Save the HTML file
with open("templates/index.html", "w") as f:
    f.write(index_html)
