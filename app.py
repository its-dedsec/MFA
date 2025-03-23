import streamlit as st
import numpy as np
import pickle
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

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

st.title("Keystroke Dynamics-Based MFA")
st.write("Enter your keystrokes for authentication")

keystroke_input = st.text_input("Type a sample password")

if st.button("Train Model"):
    if len(keystroke_input) < 5:
        st.error("Need more samples to train")
    else:
        X = np.array([list(map(ord, keystroke_input))])
        model.fit(X)
        trained = True
        
        with open("keystroke_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Visualization: Show training progress without sleep
        progress_bar = st.progress(0)
        for i in range(1, 101):
            progress_bar.progress(i)
        st.success("Training Completed!")

if st.button("Verify User"):
    if not trained:
        st.error("Model not trained yet")
    else:
        X_test = np.array([list(map(ord, keystroke_input))])
        prediction = model.predict(X_test)
        if prediction[0] == 1:
            st.success("Authenticated")
        else:
            st.error("Verification failed, use secondary MFA")
        
        # Visualization: Plot keystroke data
        fig, ax = plt.subplots()
        ax.plot(range(len(keystroke_input)), list(map(ord, keystroke_input)), marker='o', linestyle='-')
        ax.set_title("Keystroke Data Processing")
        ax.set_xlabel("Keystroke Index")
        ax.set_ylabel("ASCII Value")
        st.pyplot(fig)
