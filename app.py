import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import time
import hashlib
import plotly.express as px
import os  # Import the os module

# Page configuration
st.set_page_config(page_title="Keystroke Dynamics-Based MFA", layout="wide")

# Initialize session state variables
if "sample_count" not in st.session_state:
    st.session_state.sample_count = 0
if "key_press_times" not in st.session_state:
    st.session_state.key_press_times = {}
if "training_data" not in st.session_state:
    st.session_state.training_data = pd.DataFrame(columns=["KeyCode", "TimeDiff", "Password"])
if "trained" not in st.session_state:
    st.session_state.trained = False
if "reference_password" not in st.session_state:
    st.session_state.reference_password = "password123"
if "samples" not in st.session_state:
    st.session_state.samples = []
if "input_method" not in st.session_state:  # Added to track input method
    st.session_state.input_method = "primary"
if "password_input_main" not in st.session_state:  # Add this
    st.session_state.password_input_main = ""
if "alt_password_input" not in st.session_state: # Add this
    st.session_state.alt_password_input = ""
if "verify_password_main" not in st.session_state: # Add this
    st.session_state.verify_password_main = ""

# Constants
DATA_FILE = "keystroke_data.csv"
MODEL_FILE = "keystroke_model.pkl"

# Load existing model if available
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
        st.session_state.trained = True
except FileNotFoundError:
    model = IsolationForest(contamination=0.1, random_state=42)
    st.session_state.trained = False

# Load existing training data if available
try:
    if os.path.exists(DATA_FILE):
        st.session_state.training_data = pd.read_csv(DATA_FILE)
except Exception:
    pass

# Function to extract features from keystroke timings
def extract_features(timings, password):
    features = []
    
    # Convert timings dict to sorted list of (key, time) tuples
    timing_list = sorted([(k, v) for k, v in timings.items()], key=lambda x: x[1])
    
    # Calculate time differences between consecutive keystrokes
    for i in range(1, len(timing_list)):
        key_code = ord(password[i])
        time_diff = timing_list[i][1] - timing_list[i - 1][1]
        
        features.append({
            "KeyCode": key_code,
            "TimeDiff": time_diff,
            "Password": hashlib.sha256(password.encode()).hexdigest()[:8],
        })
    
    return pd.DataFrame(features)


# Streamlit UI
st.title("Keystroke Dynamics-Based MFA")
st.write("This system authenticates users based on their unique typing patterns.")

# Tabs for different functionality
tab1, tab2, tab3 = st.tabs(["Training", "Authentication", "Visualization"])

# Sidebar
st.sidebar.header("Keystroke Authentication")
st.sidebar.write(f"Samples collected: {st.session_state.sample_count}/5")
st.sidebar.write("Status: " + ("Trained ✅" if st.session_state.trained else "Not Trained ❌"))
st.sidebar.write("\n\n")
st.sidebar.write("Developed for Final Year Project")

# Training Tab
with tab1:
    st.header("Training Mode")
    
    # Set reference password
    ref_pwd = st.text_input("Set Reference Password", value=st.session_state.reference_password)
    if ref_pwd != st.session_state.reference_password:
        st.session_state.reference_password = ref_pwd
        st.session_state.sample_count = 0
        st.session_state.samples = []
        st.session_state.training_data = pd.DataFrame(columns=["KeyCode", "TimeDiff", "Password"])
        st.info("Reference password updated. Please collect new samples.")
    
    st.write(f"Please type: **{st.session_state.reference_password}**")
    
    # Create columns for input and buttons
    col1, col2 = st.columns([3, 1])
    
    # Modified approach - Use regular input and handle submission via a button
    password_input = col1.text_input("Type the reference password:", key="password_input_main")
    
    # Capture keystroke timings manually
    if password_input:
        current_time = time.time()
        key_index = len(password_input) - 1
        
        # Record key press time if it's not already recorded
        if key_index >= 0 and key_index not in st.session_state.key_press_times:
            st.session_state.key_press_times[key_index] = current_time
    
    # Submit button (outside of any form now)
    if col2.button("Submit Sample", key="submit_sample_main", use_container_width=True):  # Place button in col2
        if password_input == st.session_state.reference_password:
            # Ensure we have enough keystroke data
            if len(st.session_state.key_press_times) > 1:
                # Store the sample
                st.session_state.samples.append({
                    "password": password_input,
                    "timings": st.session_state.key_press_times.copy(),
                })
                
                # Extract features
                features = extract_features(st.session_state.key_press_times, password_input)
                
                # Add to training data
                if not features.empty:
                    st.session_state.training_data = pd.concat(
                        [st.session_state.training_data, features], ignore_index=True
                    )
                
                # Increment sample count
                st.session_state.sample_count += 1
                
                st.success(f"Sample {st.session_state.sample_count} captured successfully!")
                
                # Reset timings for next sample
                st.session_state.key_press_times = {}
                # Clear input
                st.session_state.password_input_main = ""
            else:
                st.warning("Not enough keystroke data captured. Please type the password again.")
        else:
            st.warning("Please type the correct reference password")
    
    # Alternative manual input method (fallback)
    st.subheader("Alternative Method")
    st.write("If the above method doesn't work, use this manual input:")
    
    alt_password_input = st.text_input("Type the reference password:", key="alt_password_input")
    
    # Capture keystroke timings for alternative method
    if alt_password_input:
        current_time = time.time()
        alt_key_index = len(alt_password_input) - 1
        
        # Record key press time
        if alt_key_index >= 0 and alt_key_index not in st.session_state.get("alt_key_press_times", {}):
            if "alt_key_press_times" not in st.session_state:
                st.session_state.alt_key_press_times = {}
            st.session_state.alt_key_press_times[alt_key_index] = current_time
    
    # Manual submit button
    if st.button("Submit Sample (Manual)", use_container_width=True):
        if alt_password_input == st.session_state.reference_password:
            # Ensure we have alternative keystroke timings
            if "alt_key_press_times" in st.session_state and len(st.session_state.alt_key_press_times) > 1:
                # Store the sample
                st.session_state.samples.append({
                    "password": alt_password_input,
                    "timings": st.session_state.alt_key_press_times.copy(),
                })
                
                # Extract features
                features = extract_features(st.session_state.alt_key_press_times, alt_password_input)
                
                # Add to training data
                if not features.empty:
                    st.session_state.training_data = pd.concat(
                        [st.session_state.training_data, features], ignore_index=True
                    )
                
                # Increment sample count
                st.session_state.sample_count += 1
                
                st.success(f"Sample {st.session_state.sample_count} captured successfully!")
                
                # Reset timings for next sample
                st.session_state.alt_key_press_times = {}
                st.session_state.alt_password_input = "" # Clear
            else:
                st.warning("Not enough keystroke data captured. Please type the password again.")
        else:
            st.warning("Please type the correct reference password")
    
    # Display sample progress
    if st.session_state.sample_count > 0:
        st.progress(min(st.session_state.sample_count / 5, 1.0))
    
    # Train model button
    if st.button("Train Model", use_container_width=True):
        if st.session_state.sample_count < 3:
            st.error("Need at least 3 samples to train. Please collect more samples.")
        else:
            # Show progress bar
            progress_bar = st.progress(0)
            
            # Train model
            try:
                X = st.session_state.training_data[["KeyCode", "TimeDiff"]].values
                
                # Update progress
                for i in range(50):
                    progress_bar.progress(i / 100)
                    time.sleep(0.01)
                
                # Train the model
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X)
                
                # Update progress
                for i in range(50, 101):
                    progress_bar.progress(i / 100)
                    time.sleep(0.01)
                
                # Save model and data
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(model, f)
                
                st.session_state.training_data.to_csv(DATA_FILE, index=False)
                
                st.session_state.trained = True
                st.success("Training Completed! You can now verify users.")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")


# Authentication Tab
with tab2:
    st.header("Authentication Mode")
    
    if not st.session_state.trained:
        st.warning("Model not trained yet. Please train the model first.")
    else:
        st.write(f"Please type: **{st.session_state.reference_password}**")
        
        # Create columns for input and verification
        auth_col1, auth_col2 = st.columns([3, 1])
        
        # Simplified authentication approach without forms
        verify_password = auth_col1.text_input("Type password for verification:", key="verify_password_main")
        
        # Capture verification keystroke timings
        if verify_password:
            current_time = time.time()
            verify_key_index = len(verify_password) - 1
            
            # Record key press time
            if verify_key_index >= 0 and verify_key_index not in st.session_state.get("verify_key_times", {}):
                if "verify_key_times" not in st.session_state:
                    st.session_state.verify_key_times = {}
                st.session_state.verify_key_times[verify_key_index] = current_time
        
        # Verify button
        verify_button = auth_col2.button("Verify User", use_container_width=True)
        
        # Alternative manual verification
        st.subheader("Alternative Verification Method")
        alt_verify_password = st.text_input("Type the reference password to verify:", key="alt_verify_password")
        
        # Capture verification keystroke timings for alternative method
        if alt_verify_password:
            current_time = time.time()
            alt_verify_key_index = len(alt_verify_password) - 1
            
            # Record key press time
            if alt_verify_key_index >= 0 and alt_verify_key_index not in st.session_state.get(
                "alt_verify_key_times", {}
            ):
                if "alt_verify_key_times" not in st.session_state:
                    st.session_state.alt_verify_key_times = {}
                st.session_state.alt_verify_key_times[alt_verify_key_index] = current_time
        
        manual_verify = st.button("Verify (Manual)", use_container_width=True)
        
        # Process verification
        if verify_button or manual_verify:
            try:
                # Get the password and timings
                if verify_button:
                    verify_timings = st.session_state.get("verify_key_times", {})
                    password = verify_password
                elif manual_verify:
                    verify_timings = st.session_state.get("alt_verify_key_times", {})
                    password = alt_verify_password
                else:
                    verify_timings = {}
                    password = ""
                
                # Verify the user
                if password == st.session_state.reference_password and len(verify_timings) > 1:
                    # Show progress
                    auth_progress = st.progress(0)
                    for i in range(50):
                        auth_progress.progress(i / 100)
                        time.sleep(0.01)
                    
                    # Extract features
                    features = extract_features(verify_timings, password)
                    
                    if not features.empty:
                        # Prepare test data
                        X_test = features[["KeyCode", "TimeDiff"]].values
                        
                        # Make prediction
                        predictions = model.predict(X_test)
                        scores = model.decision_function(X_test)
                        
                        # Calculate authentication score
                        auth_score = np.mean(scores)
                        
                        # Determine if authenticated
                        is_authenticated = np.mean(predictions) == 1  # 1 means inlier (normal)
                        
                        # Update progress
                        for i in range(50, 101):
                            auth_progress.progress(i / 100)
                            time.sleep(0.01)
                        
                        # Display result
                        if is_authenticated:
                            st.success(f"✅ Authentication successful! (Score: {auth_score:.2f})")
                            st.balloons()
                        else:
                            st.error(f"❌ Authentication failed. (Score: {auth_score:.2f})")
                            st.warning(
                                "Unusual typing pattern detected. Please try again or use alternative authentication."
                            )
                    else:
                        st.error("Not enough data to verify. Please type the complete password.")
                else:
                    st.warning("Please type the correct reference password completely.")
                
                # Reset timings for next verification
                if "verify_key_times" in st.session_state:
                    st.session_state.verify_key_times = {}
                if "alt_verify_key_times" in st.session_state:
                    st.session_state.alt_verify_key_times = {}
            except Exception as e:
                st.error(f"Verification error: {str(e)}")


# Visualization Tab
with tab3:
    st.header("Keystroke Data Visualization")
    
    if st.session_state.training_data.empty:
        st.info("No training data available. Please collect samples first.")
    else:
        # Data summary
        st.subheader("Training Data Summary")
        st.write(f"Total samples: {st.session_state.sample_count}")
        st.write(f"Feature records: {len(st.session_state.training_data)}")
        
        # Visualization - Typing Pattern
        st.subheader("Keystroke Timing Analysis")
        
        # Create interactive visualizations with plotly
        try:
            fig = px.scatter(
                st.session_state.training_data,
                x="KeyCode",
                y="TimeDiff",
                title="Keystroke Timing Patterns",
                labels={"KeyCode": "Key Code (ASCII)", "TimeDiff": "Time Difference (s)"},
                color="TimeDiff",
                size="TimeDiff",
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")
        
        # Distribution of time differences
        try:
            fig = px.histogram(
                st.session_state.training_data,
                x="TimeDiff",
                title="Distribution of Time Differences Between Keystrokes",
                labels={"TimeDiff": "Time Difference (seconds)"},
                nbins=20,
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating histogram: {e}")
        
        # Heatmap using matplotlib
        st.subheader("Keystroke Data Correlation Heatmap")
        
        # Create correlation matrix and heatmap
        if len(st.session_state.training_data.columns) >= 2:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                corr_matrix = st.session_state.training_data[["KeyCode", "TimeDiff"]].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Keystroke Data Correlation Heatmap")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")
