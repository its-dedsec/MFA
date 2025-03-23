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
def initialize_session_state():
    """Initializes all session state variables."""
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
    if "input_method" not in st.session_state:
        st.session_state.input_method = "primary"
    if "password_input_main" not in st.session_state:
        st.session_state.password_input_main = ""
    if "verify_password_main" not in st.session_state:
        st.session_state.verify_password_main = ""
    if "feedback" not in st.session_state:
        st.session_state.feedback = ""
    if "verify_key_times" not in st.session_state:
        st.session_state.verify_key_times = {}

initialize_session_state()  # Call the function to ensure initialization

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
    """Extracts features (KeyCode, TimeDiff) from keystroke timings."""
    features = []
    try:
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
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error
    
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
    
    st.write(
        f"Please type the password **{st.session_state.reference_password}** in the text box below and then click the 'Submit Sample' button."
    )
    
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
    
    # Submit button
    if col2.button("Submit Sample", key="submit_sample_main", use_container_width=True):
        try:
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
                    st.session_state.feedback = (
                        f"Sample {st.session_state.sample_count} captured successfully!"
                    )
                    
                    # Reset timings
                    st.session_state.key_press_times = {}
                    # DO NOT CLEAR password_input_main here.  The user should see what they typed.
                    
                else:
                    st.session_state.feedback = (
                        "Not enough keystroke data captured. Please type the password again."
                    )
            else:
                st.session_state.feedback = "Please type the correct reference password"
                st.warning("Please type the correct reference password")
        except Exception as e:
            st.error(f"Error processing sample: {e}")
            st.session_state.feedback = "An error occurred. Please try again."

    # Display feedback to the user
    st.text(st.session_state.feedback)
    
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
        st.write(
            f"Please type the password **{st.session_state.reference_password}** in the text box below and then click the 'Verify User' button."
        )
        
        # Create columns for input and verification
        auth_col1, auth_col2 = st.columns([3, 1])
        
        # Simplified authentication approach
        verify_password = auth_col1.text_input("Type password for verification:", key="verify_password_main")
        
        # Capture verification keystroke timings
        if verify_password:
            current_time = time.time()
            verify_key_index = len(verify_password) - 1
            
            # Record key press time
            if verify_key_index >= 0 and verify_key_index not in st.session_state.get("verify_key_times", {}):
                st.session_state.verify_key_times[verify_key_index] = current_time
        
        # Verify button
        verify_button = auth_col2.button("Verify User", use_container_width=True)
        
        # Process verification
        if verify_button:
            try:
                # Get the password and timings
                password = verify_password
                verify_timings = st.session_state.get("verify_key_times", {})
                
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
                                "Unusual typing pattern detected. Please try again."
                            )
                    else:
                        st.error("Not enough data to verify. Please type the complete password.")
                else:
                    st.warning("Please type the correct reference password completely.")
                
                # Reset timings and input
                st.session_state.verify_key_times = {}
                #  DO NOT CLEAR verify_password_main here.  The user should see what they typed.
                
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
