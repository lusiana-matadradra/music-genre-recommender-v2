import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Music Genre Recommender", layout="centered")
st.title("🎧 Music Genre Recommender")
st.markdown("Get a recommended genre based on your MBTI type and tempo preference.")

# Load dataset directly
df = pd.read_excel("music_preferences.xlsx", sheet_name="Form Responses 1")

# Preprocessing
tempo_map = {'Slow/Calm': 1, 'Medium': 2, 'Fast/Energetic': 3}
df['Tempo_Ordinal'] = df['What tempo of music do you prefer?'].map(tempo_map)

def get_mbti(x):
    if "Extraversion" in x:
        return "E"
    elif "Introversion" in x:
        return "I"
    return "?"

df['MBTI'] = df['When it comes to socialising:'].apply(get_mbti)

df = df[['MBTI', 'Tempo_Ordinal', 'What genre do you listen to most often?']].dropna()
le_genre = LabelEncoder()
df['Genre_Label'] = le_genre.fit_transform(df['What genre do you listen to most often?'])

# Train the model right here
X = pd.get_dummies(df[['MBTI', 'Tempo_Ordinal']])
y = df['Genre_Label']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# User input
personality = st.selectbox("Select your MBTI type:", ['E', 'I'])
tempo = st.radio("Preferred music tempo:", ['Slow/Calm', 'Medium', 'Fast/Energetic'])
tempo_val = tempo_map[tempo]

# Prepare input
input_dict = {'MBTI_E': 0, 'MBTI_I': 0}
input_dict[f'MBTI_{personality}'] = 1
input_dict['Tempo_Ordinal'] = tempo_val
input_df = pd.DataFrame([input_dict])

# Predict
if st.button("🎵 Recommend Genre"):
    pred = model.predict(input_df)[0]
    genre_name = le_genre.inverse_transform([pred])[0]
    st.success(f"🎶 Your recommended genre is: **{genre_name}**")
