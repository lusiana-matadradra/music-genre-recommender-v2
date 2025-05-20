import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Music Genre Recommender", layout="centered")
st.title("🎧 Music Genre Recommender")
st.markdown("Answer a few personality questions and we’ll recommend a music genre **and** some artists you might love!")

# Load data
df = pd.read_excel("music_preferences.xlsx", sheet_name="Form Responses 1")

# Map tempo to ordinal
tempo_map = {'Slow/Calm': 1, 'Medium': 2, 'Fast/Energetic': 3}
df['Tempo_Ordinal'] = df['What tempo of music do you prefer?'].map(tempo_map)

# Determine MBTI
def determine_mbti(row):
    mbti = ""
    mbti += "E" if "Extraversion" in row["When it comes to socialising:"] else "I"
    mbti += "S" if "Sensing" in row["When processing information:"] else "N"
    mbti += "T" if "Thinking" in row["When making decisions:"] else "F"
    mbti += "J" if "Judging" in row["When planning my day or tasks:"] else "P"
    return mbti

df['MBTI'] = df.apply(determine_mbti, axis=1)
df = df[['MBTI', 'Tempo_Ordinal', 'What genre do you listen to most often?']].dropna()

# Encode genre labels
le_genre = LabelEncoder()
df['Genre_Label'] = le_genre.fit_transform(df['What genre do you listen to most often?'])

# Train model
X = pd.get_dummies(df[['MBTI', 'Tempo_Ordinal']])
y = df['Genre_Label']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Artist mapping
genre_artists = {
    'Rap': "ASAP Rocky, Travis Scott, Drake, Kendrick Lamar",
    'Pop': "Taylor Swift, Beyonce, Michael Jackson, Sabrina Carpenter",
    'Rock': "Green Day, The Beatles, Nirvana, Guns 'n Roses",
    'R&B': "Chris Brown, Daniel Caesar, Brent Faiyaz, Frank Ocean",
    'Indie': "Clairo, Arctic Monkeys, Phoebe Bridgers, Tame Impala",
    'Classical': "Mozart, Bach, Beethoven",
    'Jazz': "Miles Davis, John Coltrane, Ella Fitzgerald"
}

# --- User Questions ---
st.header("🧠 Tell us about yourself")

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

tempo = st.radio("Preferred music tempo:", ['Slow/Calm', 'Medium', 'Fast/Energetic'])
tempo_val = tempo_map[tempo]

# Build MBTI
mbti = ""
mbti += "E" if "Extraversion" in social else "I"
mbti += "S" if "Sensing" in info else "N"
mbti += "T" if "Thinking" in decisions else "F"
mbti += "J" if "Judging" in planning else "P"

# Create input DataFrame
input_dict = {col: 0 for col in X.columns}
input_dict['Tempo_Ordinal'] = tempo_val
input_dict[f'MBTI_{mbti}'] = 1
input_df = pd.DataFrame([input_dict])

# Align column order
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

# Predict
if st.button("🎵 Recommend Genre"):
    pred = model.predict(input_df)[0]
    genre = le_genre.inverse_transform([pred])[0]
    artist = genre_artists.get(genre, "a mix of great artists")
    st.success(f"🎶 Based on your answers, we recommend: **{genre}**")
    st.info(f"✨ We also think you'd love: **{artist}**")
