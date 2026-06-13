import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIG (PREMIUM LOOK)
# -----------------------------
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (PREMIUM UI)
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00ffcc;
}

/* Glass card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00ffcc, #00b3ff);
    color: black;
    font-weight: bold;
    padding: 10px;
    border-radius: 12px;
    border: none;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
model = joblib.load("fitness_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("⚡ AI FITNESS PANEL")
    st.write("Adjust your workout settings")

    st.info("💡 Tip: Train consistently for best results 💪")

    st.markdown("---")
    st.write("Built with ❤️ using Streamlit")

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="title">💪 AI FITNESS COACH</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# INPUT SECTION (COLUMNS)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    bodypart = st.selectbox("💪 Body Part", ["chest","back","legs","arms","shoulders","abs"])
    level = st.selectbox("🏋️ Level", ["beginner","intermediate","advanced"])

with col2:
    equipment = st.selectbox("🏋️ Equipment", ["bodyweight","dumbbell","barbell","machine"])
    rating = st.slider("⚡ Intensity", 1, 5, 3)

st.write("")

# -----------------------------
# BUTTON + PREDICTION
# -----------------------------
if st.button("🚀 Generate AI Workout Plan"):

    try:
        input_data = pd.DataFrame([[bodypart, equipment, level, rating]],
                                  columns=["BodyPart","Equipment","Level","Rating"])

        # encoding
        for col in ["BodyPart","Equipment","Level"]:
            input_data[col] = input_data[col].astype("category").cat.codes

        scaled = scaler.transform(input_data)

        pred = model.predict(scaled)
        workout_type = label_encoder.inverse_transform(pred)[0]

        # -----------------------------
        # OUTPUT UI (PREMIUM CARDS)
        # -----------------------------
        st.markdown("---")
        st.success("🎯 AI ANALYSIS COMPLETE")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("🏋️ Workout Type", workout_type)

        with c2:
            st.metric("🔥 Intensity", rating)

        with c3:
            st.metric("📊 Level", level)

        st.markdown("---")

        st.markdown("## 💪 Recommended Exercises")

        suggestions = {
            "chest": ["Bench Press", "Push-ups", "Chest Fly"],
            "back": ["Pull-ups", "Lat Pulldown", "Rows"],
            "legs": ["Squats", "Lunges", "Leg Press"],
            "arms": ["Bicep Curl", "Tricep Dips", "Hammer Curl"],
            "shoulders": ["Shoulder Press", "Lateral Raise", "Front Raise"],
            "abs": ["Plank", "Crunches", "Leg Raises"]
        }

        ex = suggestions.get(bodypart, ["Jumping Jacks", "Burpees", "Plank"])

        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("### 🔥 Exercise 1")
            st.write(ex[0])

        with colB:
            st.markdown("### 🔥 Exercise 2")
            st.write(ex[1])

        with colC:
            st.markdown("### 🔥 Exercise 3")
            st.write(ex[2])

        st.markdown("---")

        st.info("⚡ Tip: Consistency beats intensity. Train smart 💯")

    except Exception as e:
        st.error(f"Error: {e}")