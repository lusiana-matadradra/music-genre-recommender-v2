import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("🎧 Music Genre Recommender")
st.markdown("Get a recommended genre based on your MBTI type and tempo preference.")

# Load model
model = joblib.load("model.pkl")

# Input
personality = st.selectbox("Select your MBTI type:", ['E', 'I'])
tempo = st.radio("Preferred music tempo:", ['Slow/Calm', 'Medium', 'Fast/Energetic'])
tempo_map = {'Slow/Calm': 1, 'Medium': 2, 'Fast/Energetic': 3}
tempo_val = tempo_map[tempo]

# Predict
if st.button("Recommend Genre"):
    # Encode MBTI
    input_dict = {'MBTI_E': 0, 'MBTI_I': 0}
    input_dict[f'MBTI_{personality}'] = 1
    input_dict['Tempo_Ordinal'] = tempo_val
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    st.success(f"🎶 Your recommended genre label is: **{prediction}**")
