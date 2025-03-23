import streamlit as st
import numpy as np
import pickle
from sklearn.svm import OneClassSVM

# Initialize session state
if "model" not in st.session_state:
    st.session_state["model"] = OneClassSVM(kernel='rbf', gamma='auto')
    st.session_state["trained"] = False
    st.session_state["keystroke_data"] = []

st.title("Keystroke Dynamics-Based MFA")
st.write("Enter your keystrokes for authentication")

# Keystroke data collection
keystroke_data = st.text_input("Type a sample password")

if st.button("Train Model"):
    if len(keystroke_data) < 5:
        st.error("Need more samples to train!")
    else:
        # Convert to numerical features
        features = np.array([list(map(ord, keystroke_data))])
        st.session_state["model"].fit(features)
        st.session_state["trained"] = True
        st.success("Model trained successfully!")

if st.button("Verify User"):
    if not st.session_state["trained"]:
        st.error("Model is not trained yet!")
    else:
        # Convert to numerical features
        test_features = np.array([list(map(ord, keystroke_data))])
        prediction = st.session_state["model"].predict(test_features)
        
        if prediction[0] == 1:
            st.success("✅ Authenticated")
        else:
            st.error("❌ Verification failed! Use secondary MFA")

