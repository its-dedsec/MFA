import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from datetime import datetime
import os
import hashlib
import time
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Keystroke Dynamics-Based MFA", layout="wide")

# Initialize session state variables
if "keystroke_timings" not in st.session_state:
    st.session_state.keystroke_timings = []
if "current_typing" not in st.session_state:
    st.session_state.current_typing = ""
if "prev_time" not in st.session_state:
    st.session_state.prev_time = None
if "key_times" not in st.session_state:
    st.session_state.key_times = []
if "reference_password" not in st.session_state:
    st.session_state.reference_password = "password123"  # Default reference password
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "training_data" not in st.session_state:
    st.session_state.training_data = pd.DataFrame(columns=["KeyPair", "TimeDiff", "TrigramTime", "Password"])

# Constants
DATA_FILE = "keystroke_data.csv"
MODEL_FILE = "keystroke_model.pkl"

# Function to extract features from keystroke data
def extract_features(key_times, password):
    features = []
    
    # Skip if not enough data
    if len(key_times) < 2:
        return pd.DataFrame()
    
    # Extract key pair timing features
    for i in range(len(key_times) - 1):
        key_pair = f"{password[i]}{password[i+1]}"
        time_diff = key_times[i+1] - key_times[i]
        
        # Add trigram timing if possible
        trigram_time = 0
        if i < len(key_times) - 2:
            trigram_time = key_times[i+2] - key_times[i]
        
        features.append({
            "KeyPair": hash(key_pair),  # Hash to anonymize actual characters
            "TimeDiff": time_diff,
            "TrigramTime": trigram_time,
            "Password": hashlib.sha256(password.encode()).hexdigest()[:8]  # Short hash of password
        })
    
    return pd.DataFrame(features)

# Function to capture keystroke timings
def capture_keystroke(password):
    current_time = time.time()
    
    # Record times for each keystroke
    if len(password) > len(st.session_state.current_typing):
        # A new key was pressed
        st.session_state.key_times.append(current_time)
    elif len(password) < len(st.session_state.current_typing):
        # A key was deleted
        if st.session_state.key_times:
            st.session_state.key_times.pop()
    
    st.session_state.current_typing = password
    
    # Check if complete password was entered
    if password == st.session_state.reference_password and len(st.session_state.key_times) == len(password):
        # Complete password entered, add to samples
        st.session_state.keystroke_timings.append({
            "password": password,
            "key_times": st.session_state.key_times.copy()
        })
        st.success(f"Sample {len(st.session_state.keystroke_timings)} captured!")
        
        # Extract features and add to training data
        features_df = extract_features(st.session_state.key_times, password)
        if not features_df.empty:
            st.session_state.training_data = pd.concat([st.session_state.training_data, features_df], ignore_index=True)
        
        # Reset for next sample
        st.session_state.current_typing = ""
        st.session_state.key_times = []
        return True
    
    return False

# Function to train the model
def train_model():
    if len(st.session_state.keystroke_timings) < 3:
        st.error("Need at least 3 samples to train the model.")
        return False
    
    # Prepare features for training
    X = st.session_state.training_data[["KeyPair", "TimeDiff", "TrigramTime"]].values
    
    # Train model (OneClassSVM for anomaly detection)
    model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
    model.fit(X)
    
    # Save model
    st.session_state.model = model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    # Save training data
    st.session_state.training_data.to_csv(DATA_FILE, index=False)
    
    st.session_state.trained = True
    return True

# Function to verify a user
def verify_user(password, key_times):
    if not st.session_state.trained or st.session_state.model is None:
        st.error("Model not trained yet. Please train the model first.")
        return False, 0
    
    # Extract features from current typing pattern
    features_df = extract_features(key_times, password)
    if features_df.empty:
        st.error("Not enough keystroke data to verify.")
        return False, 0
    
    # Prepare test data
    X_test = features_df[["KeyPair", "TimeDiff", "TrigramTime"]].values
    
    # Get predictions and decision scores
    predictions = st.session_state.model.predict(X_test)
    scores = st.session_state.model.decision_function(X_test)
    
    # Calculate authentication score (average of decision scores)
    auth_score = np.mean(scores)
    
    # Determine if authenticated
    is_authenticated = np.mean(predictions) == 1  # For IsolationForest, 1 means inlier (normal)
    
    return is_authenticated, auth_score

# Try to load existing model and data
try:
    with open(MODEL_FILE, "rb") as f:
        st.session_state.model = pickle.load(f)
        st.session_state.trained = True
    
    if os.path.exists(DATA_FILE):
        st.session_state.training_data = pd.read_csv(DATA_FILE)
except Exception as e:
    st.session_state.trained = False
    st.session_state.model = None

# UI Layout
st.title("Keystroke Dynamics-Based MFA")
st.write("This system authenticates users based on their unique typing patterns.")

# Sidebar
st.sidebar.header("Keystroke Authentication")
st.sidebar.write(f"Samples collected: {len(st.session_state.keystroke_timings)} / 5")
st.sidebar.write("Status: " + ("Trained ✅" if st.session_state.trained else "Not Trained ❌"))

# Tabs for different functionality
tab1, tab2, tab3 = st.tabs(["Training", "Authentication", "Visualization"])

# Training Tab
with tab1:
    st.header("Training Mode")
    
    # Set reference password
    new_ref_password = st.text_input("Set Reference Password (will be used for both training and verification)", 
                                     value=st.session_state.reference_password)
    if new_ref_password != st.session_state.reference_password:
        st.session_state.reference_password = new_ref_password
        st.session_state.keystroke_timings = []  # Reset samples if password changes
        st.session_state.training_data = pd.DataFrame(columns=["KeyPair", "TimeDiff", "TrigramTime", "Password"])
        st.info("Reference password updated. Please collect new samples.")
    
    st.write(f"Type the reference password: **{st.session_state.reference_password}**")
    
    # Password input for training
    train_password = st.text_input("Enter password for training:", key="train_input")
    
    # Capture keystroke timing
    if train_password:
        sample_complete = capture_keystroke(train_password)
    
    # Display sample progress
    if st.session_state.keystroke_timings:
        st.progress(min(len(st.session_state.keystroke_timings)/5, 1.0))
    
    # Train model button
    if st.button("Train Model"):
        progress = st.progress(0)
        for i in range(1, 101):
            # Simulate training progress
            progress.progress(i/100)
            time.sleep(0.01)
        
        if train_model():
            st.success("Model trained successfully! You can now verify your identity.")

# Authentication Tab
with tab2:
    st.header("Authentication Mode")
    
    st.write(f"Type the reference password: **{st.session_state.reference_password}**")
    
    # Password input for verification
    verify_password = st.text_input("Enter password to verify your identity:", key="verify_input")
    
    # Verify button
    if st.button("Verify Identity"):
        if not verify_password:
            st.warning("Please enter the password first.")
        elif verify_password != st.session_state.reference_password:
            st.error("Incorrect password. Please enter the correct password.")
        elif len(st.session_state.key_times) != len(verify_password):
            st.warning("Please type the complete password.")
        else:
            # Show verification progress
            progress = st.progress(0)
            for i in range(1, 101):
                progress.progress(i/100)
                time.sleep(0.01)
            
            # Verify user
            is_authenticated, auth_score = verify_user(verify_password, st.session_state.key_times)
            
            # Display result
            if is_authenticated:
                st.success(f"✅ Authentication successful! (Score: {auth_score:.2f})")
                
                # Add confetti effect for successful authentication
                st.balloons()
            else:
                st.error(f"❌ Authentication failed. (Score: {auth_score:.2f})")
                st.warning("Unusual typing pattern detected. Please try again or use alternative authentication.")

# Visualization Tab
with tab3:
    st.header("Visualizations")
    
    if st.session_state.training_data.empty:
        st.info("No data available for visualization. Please collect samples first.")
    else:
        # Data summary
        st.subheader("Training Data Summary")
        st.write(f"Total samples: {len(st.session_state.keystroke_timings)}")
        st.write(f"Feature records: {len(st.session_state.training_data)}")
        
        # Timing distribution visualization
        st.subheader("Keystroke Timing Distribution")
        
        # Use plotly for interactive visualization
        fig = px.histogram(st.session_state.training_data, x="TimeDiff", 
                          title="Distribution of Time Differences Between Keystrokes",
                          labels={"TimeDiff": "Time Difference (seconds)"},
                          nbins=20)
        st.plotly_chart(fig)
        
        # Scatter plot of timing patterns
        st.subheader("Keystroke Pattern Visualization")
        
        fig = px.scatter(st.session_state.training_data, x="KeyPair", y="TimeDiff",
                        color="TrigramTime", size="TimeDiff",
                        title="Keystroke Timing Patterns",
                        labels={"KeyPair": "Key Combination", 
                               "TimeDiff": "Time Between Keystrokes (s)",
                               "TrigramTime": "Trigram Timing (s)"})
        st.plotly_chart(fig)
        
        # Show heatmap
        st.subheader("Correlation Heatmap")
        
        # Create correlation matrix and heatmap
        corr_matrix = st.session_state.training_data[["KeyPair", "TimeDiff", "TrigramTime"]].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Keystroke Dynamics-Based MFA")
st.sidebar.write("Final Year Project")
