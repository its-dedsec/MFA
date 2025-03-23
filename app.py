import streamlit as st
import numpy as np
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from datetime import datetime
import os

# Load or initialize model
try:
    with open("keystroke_model.pkl", "rb") as f:
        model = pickle.load(f)
        trained = True
except FileNotFoundError:
    model = OneClassSVM(kernel='rbf', gamma='auto')
    trained = False

keystroke_timings = []  # Store timing data
DATA_FILE = "keystroke_data.csv"

# Load existing keystroke data
if os.path.exists(DATA_FILE):
    keystroke_data = pd.read_csv(DATA_FILE)
else:
    keystroke_data = pd.DataFrame(columns=["ASCII Value", "Time Difference"])

# Streamlit UI
st.set_page_config(page_title="Keystroke Dynamics-Based MFA", layout="wide")
st.sidebar.header("Keystroke Authentication")
st.title("Keystroke Dynamics-Based MFA")
st.write("This system authenticates users based on their unique typing patterns.")

# Input
keystroke_input = st.text_input("Type a sample password")

# Capture timing data
if keystroke_input:
    keystroke_timings.append((keystroke_input, datetime.now()))

# Show sample count
st.sidebar.write(f"Samples collected: {len(keystroke_timings)} / 5")

# Train Model
if st.button("Train Model"):
    if len(keystroke_timings) < 5:
        st.error("Need at least 5 samples to train. Keep typing and try again.")
    else:
        df = pd.DataFrame([(ord(char), (t2 - t1).total_seconds()) 
                           for (text, t1), (text2, t2) in zip(keystroke_timings[:-1], keystroke_timings[1:])], 
                          columns=["ASCII Value", "Time Difference"])
        
        keystroke_data = pd.concat([keystroke_data, df], ignore_index=True)
        keystroke_data.to_csv(DATA_FILE, index=False)
        
        X = keystroke_data[["ASCII Value", "Time Difference"]].values
        model.fit(X)
        trained = True
        with open("keystroke_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Visualization - Training progress
        progress_bar = st.progress(0)
        for i in range(1, 101):
            progress_bar.progress(i)
        st.success("Training Completed!")

# Verify User
if st.button("Verify User"):
    if not trained:
        st.error("Model not trained yet")
    else:
        df_test = pd.DataFrame([(ord(char), (t2 - t1).total_seconds()) 
                                for (text, t1), (text2, t2) in zip(keystroke_timings[:-1], keystroke_timings[1:])], 
                               columns=["ASCII Value", "Time Difference"])
        X_test = df_test[["ASCII Value", "Time Difference"]].values
        prediction = model.predict(X_test)
        if np.mean(prediction) > 0:
            st.success("Authenticated")
        else:
            st.error("Verification failed, use secondary MFA")
        
        # Visualization - Keystroke Data Heatmap
        fig, ax = plt.subplots()
        sns.heatmap(keystroke_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Keystroke Data Correlation Heatmap")
        st.pyplot(fig)

        # Visualization - Typing Pattern
        fig, ax = plt.subplots()
        ax.plot(keystroke_data["ASCII Value"], keystroke_data["Time Difference"], marker='o', linestyle='-', label='Typing Pattern')
        ax.set_title("Keystroke Timing Analysis")
        ax.set_xlabel("Keystroke ASCII Value")
        ax.set_ylabel("Time Difference (s)")
        ax.legend()
        st.pyplot(fig)

st.sidebar.write("\n\n")
st.sidebar.write("Developed for Final Year Project")
