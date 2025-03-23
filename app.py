import streamlit as st
import requests

st.title("Keystroke Dynamics-Based MFA")
st.write("Enter your keystrokes for authentication")

keystroke_data = st.text_input("Type a sample password")
if st.button("Train Model"):
    response = requests.post("http://127.0.0.1:5000/train", json={"keystrokes": [list(map(ord, keystroke_data))]})
    st.write(response.json())

if st.button("Verify User"):
    response = requests.post("http://127.0.0.1:5000/verify", json={"keystroke": list(map(ord, keystroke_data))})
    st.write(response.json())
