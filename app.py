import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("🎧 Music Genre Recommender")
st.markdown("Get a recommended genre based on your **full MBTI type** and **tempo preference**.")

# Load model and label encoder
model = joblib.load("model.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

# Define list of MBTI types used during training
mbti_types = [
    'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
    'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
    'INFJ', 'INFP', 'INTJ', 'INTP',
    'ISFJ', 'ISFP', 'ISTJ', 'ISTP'
]

# User Inputs
mbti = st.selectbox("Select your MBTI type:", mbti_types)
tempo = st.radio("Preferred music tempo:", ['Slow/Calm', 'Medium', 'Fast/Energetic'])
tempo_map = {'Slow/Calm': 1, 'Medium': 2, 'Fast/Energetic': 3}
tempo_val = tempo_map[tempo]

# Create input dictionary for prediction
input_dict = {'Tempo_Ordinal': tempo_val}
for mbti_option in mbti_types:
    input_dict[f'MBTI_{mbti_option}'] = 1 if mbti == mbti_option else 0

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("🎵 Recommend Genre"):
    pred = model.predict(input_df)[0]
    genre = genre_encoder.inverse_transform([pred])[0]
    st.success(f"🎶 Based on your MBTI and tempo, we recommend: **{genre}**")
