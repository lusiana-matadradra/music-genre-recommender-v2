import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("🎧 Music Genre Recommender")
st.markdown("Get a recommended genre based on your **full MBTI type** and **tempo preference**.")

# Load dataset
df = pd.read_excel("music_preferences.xlsx", sheet_name="Form Responses 1")

# Preprocess data
tempo_map = {'Slow/Calm': 1, 'Medium': 2, 'Fast/Energetic': 3}
df['Tempo_Ordinal'] = df['What tempo of music do you prefer?'].map(tempo_map)

# Build MBTI string
def determine_mbti(row):
    mbti = ""
    mbti += "E" if "Extraversion" in row["When it comes to socialising:"] else "I"
    mbti += "S" if "Sensing" in row["When processing information:"] else "N"
    mbti += "T" if "Thinking" in row["When making decisions:"] else "F"
    mbti += "J" if "Judging" in row["When planning my day or tasks:"] else "P"
    return mbti

df['MBTI'] = df.apply(determine_mbti, axis=1)

# Drop missing and encode
df = df[['MBTI', 'Tempo_Ordinal', 'What genre do you listen to most often?']].dropna()
le_genre = LabelEncoder()
df['Genre_Label'] = le_genre.fit_transform(df['What genre do you listen to most often?'])

# One-hot encode MBTI
X = pd.get_dummies(df[['MBTI', 'Tempo_Ordinal']])
y = df['Genre_Label']

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# MBTI list
mbti_types = sorted(df['MBTI'].unique())

# --- Streamlit Input ---
user_mbti = st.selectbox("Select your MBTI type:", mbti_types)
tempo = st.radio("Preferred tempo:", ['Slow/Calm', 'Medium', 'Fast/Energetic'])
tempo_val = tempo_map[tempo]

# --- Create input vector ---
input_dict = {col: 0 for col in X.columns}
input_dict['Tempo_Ordinal'] = tempo_val
input_dict[f'MBTI_{user_mbti}'] = 1
input_df = pd.DataFrame([input_dict])

# Match column order
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

# --- Predict ---
if st.button("🎵 Recommend Genre"):
    pred = model.predict(input_df)[0]
    genre = le_genre.inverse_transform([pred])[0]
    st.success(f"🎶 Based on your MBTI and tempo, we recommend: **{genre}**")
