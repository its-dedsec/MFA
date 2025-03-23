import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.stats import iqr
import base64
import json
import os

# Function to collect keystroke data using JavaScript
def get_keystroke_data():
    keystroke_data = []
    js_code = """
        <script>
        var keystrokes = [];
        document.addEventListener("keydown", function(event) {
            keystrokes.push({
                key: event.key,
                time: Date.now()
            });
        });
        function sendKeystrokeData() {
            var data = JSON.stringify(keystrokes);
            fetch("/keystroke_data", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: data
            });
        }
        window.addEventListener("beforeunload", sendKeystrokeData);
        </script>
    """
    st.components.v1.html(js_code, height=0)
    return keystroke_data

# Function to extract features from keystroke data
def extract_features(data):
    if len(data) < 2:
        return None
    times = [event['time'] for event in data]
    intervals = np.diff(times)
    features = {
        'mean_interval': np.mean(intervals),
        'std_dev': np.std(intervals),
        'iqr': iqr(intervals),
        'max_interval': np.max(intervals),
        'min_interval': np.min(intervals)
    }
    return pd.DataFrame([features])

# Function to train model
def train_model(features):
    model = IsolationForest(contamination=0.05)
    model.fit(features)
    return model

# Load existing model or train new one
def load_or_train_model(data_file):
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            model = pickle.load(f)
    else:
        model = train_model(pd.read_csv("keystroke_data.csv"))
        with open(data_file, "wb") as f:
            pickle.dump(model, f)
    return model

# Authenticate user based on keystroke dynamics
def authenticate_user(model, new_data):
    prediction = model.predict(new_data)
    return prediction[0] == 1

# Streamlit UI
st.title("Keystroke Dynamics Authentication")
st.sidebar.header("User Options")
option = st.sidebar.radio("Choose an action:", ["Train Model", "Authenticate"])

if option == "Train Model":
    st.subheader("Collect Keystroke Data")
    data = get_keystroke_data()
    if st.button("Save Data"):
        pd.DataFrame(data).to_csv("keystroke_data.csv", index=False)
        st.success("Data saved successfully!")
        
elif option == "Authenticate":
    model = load_or_train_model("keystroke_model.pkl")
    data = get_keystroke_data()
    new_features = extract_features(data)
    if new_features is not None:
        is_authenticated = authenticate_user(model, new_features)
        st.write("Authenticated!" if is_authenticated else "Access Denied!")
