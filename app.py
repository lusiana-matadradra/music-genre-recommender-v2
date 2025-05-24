import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Music Genre Recommender", layout="centered")
st.title("\U0001F3A7 Music Genre Recommender")
st.markdown("Answer a few personality questions and we’ll recommend a **music genre** and some **artists** just for you!")

# Load dataset
df = pd.read_excel("Personality and Music Preferences.xlsx", sheet_name="Form Responses 1")

tempo_map = {'Slow/Calm': 1, 'Medium': 2, 'Fast/Energetic': 3}
df['Tempo_Ordinal'] = df['What tempo of music do you prefer?'].map(tempo_map)

def determine_mbti(row):
    mbti = ""
    mbti += "E" if "Extraversion" in row["When it comes to socialising:"] else "I"
    mbti += "S" if "Sensing" in row["When processing information:"] else "N"
    mbti += "T" if "Thinking" in row["When making decisions:"] else "F"
    mbti += "J" if "Judging" in row["When planning my day or tasks:"] else "P"
    return mbti

df['MBTI'] = df.apply(determine_mbti, axis=1)
df = df[['MBTI', 'Tempo_Ordinal', 'What genre do you listen to most often?']].dropna()

le_genre = LabelEncoder()
df['Genre_Label'] = le_genre.fit_transform(df['What genre do you listen to most often?'])

X = pd.get_dummies(df[['MBTI', 'Tempo_Ordinal']])
y = df['Genre_Label']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Artist group mapping from final dataset
genre_artists = {
    'Classical/Jazz': "Mozart, Beethoven, Miles Davis, John Coltrane",
    'Indie': "Clairo, Arctic Monkeys, Phoebe Bridgers, Tame Impala",
    'Lofi/Melody': "Lofi Girl, Eevee, Jinsang, The Deli",
    'Pop': "Taylor Swift, Beyonce, Michael Jackson, Sabrina Carpenter",
    'R&B': "Chris Brown, Daniel Caesar, Brent Faiyaz, Frank Ocean",
    'Rap': "ASAP Rocky, Travis Scott, Drake, Kendrick Lamar",
    'Rock': "Green Day, The Beatles, Nirvana, Guns N’ Roses"
}

# User Input Section
st.header("\U0001F9E0 Tell us about yourself")

social = st.selectbox("When it comes to socialising:", [
    "I prefer spending time alone or with a small group of close friends (Introversion)",
    "I enjoy large social gatherings and meeting new people (Extraversion)"
])

info = st.selectbox("When processing information:", [
    "I trust facts, data, and real experiences (Sensing)",
    "I focus on patterns, ideas, and possibilities (Intuition)"
])

decisions = st.selectbox("When making decisions:", [
    "I prioritise logic and objectivity (Thinking)",
    "I consider emotions and values (Feeling)"
])

planning = st.selectbox("When planning my day or tasks:", [
    "I like structure, planning, and sticking to schedules (Judging)",
    "I prefer being spontaneous and flexible (Perceiving)"
])

tempo = st.selectbox("Preferred music tempo:", ['Slow/Calm', 'Medium', 'Fast/Energetic'])
tempo_val = tempo_map[tempo]

mbti = ""
mbti += "E" if "Extraversion" in social else "I"
mbti += "S" if "Sensing" in info else "N"
mbti += "T" if "Thinking" in decisions else "F"
mbti += "J" if "Judging" in planning else "P"

input_dict = {col: 0 for col in X.columns}
input_dict['Tempo_Ordinal'] = tempo_val
input_dict[f'MBTI_{mbti}'] = 1
input_df = pd.DataFrame([input_dict])

for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

if st.button("\U0001F3B5 Recommend Genre"):
    pred = model.predict(input_df)[0]
    genre = le_genre.inverse_transform([pred])[0]
    artists = genre_artists.get(genre, "a mix of great artists")

    st.success(f"✨ As an **{mbti}**, you're matched with **{genre}** music!")
    st.info(f"🎤 You may enjoy artists such as: **{artists}**")
