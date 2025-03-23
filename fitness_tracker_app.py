import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/workout_fitness_tracker_data.csv")

# Load trained model and preprocessors
try:
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_gender = joblib.load("models/label_encoder_gender.pkl")
    le_workout = joblib.load("models/label_encoder_workout.pkl")
except:
    model, scaler, le_gender, le_workout = None, None, None, None

# Streamlit UI with modern styling
st.set_page_config(page_title="🏋️ Personal Fitness Tracker", layout="wide")

# Custom CSS for enhanced UI
def load_css():
    st.markdown("""
    <style>
        body {background-color: #f0f2f6;}
        .main {background-color: white; padding: 2rem; border-radius: 10px;}
        h1, h2, h3 {color: #ff5733; text-align: center;}
        .stButton>button {background-color: #ff5733; color: white; border-radius: 10px; padding: 10px;}
        .stSidebar {background-color: #2E3B55; color: white; padding: 20px;}
    </style>
    """, unsafe_allow_html=True)
load_css()

# Sidebar with expanded sections
st.sidebar.title("🎯 Enter Your Fitness Data")
st.sidebar.markdown("Use this tracker to monitor your workouts and calories burned.")

# User Input Form
with st.sidebar:
    age = st.slider("Age", 10, 100, 25)
    gender = st.selectbox("Gender", df["Gender"].unique())
    height = st.slider("Height (cm)", 120, 220, 170)
    weight = st.slider("Weight (kg)", 30, 150, 70)
    workout_type = st.selectbox("Workout Type", df["Workout Type"].unique())
    workout_duration = st.slider("Workout Duration (mins)", 5, 180, 30)
    
# Convert categorical input to numerical
gender_num = le_gender.transform([gender])[0] if le_gender else 0
workout_num = le_workout.transform([workout_type])[0] if le_workout else 0

# Prepare input for prediction
input_data = np.array([[age, gender_num, height, weight, workout_num, workout_duration]])
input_data = scaler.transform(input_data) if scaler else input_data

# Predict Calories Burned
if st.sidebar.button("🔥 Predict Calories Burned", use_container_width=True):
    if model:
        prediction = model.predict(input_data)
        st.sidebar.success(f"🔥 Estimated Calories Burned: {round(prediction[0], 2)} kcal")
    else:
        st.sidebar.warning("⚠️ Model not available. Train the model first.")

# Main Page
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.markdown("<div class='main'>Monitor your fitness, track calories, and analyze workout trends!</div>", unsafe_allow_html=True)

# Layout to fit graphs on a single screen
col1, col2 = st.columns(2)

# Graph 1: Workout Trends
with col1:
    st.subheader("📊 Workout Trends - Average Workout Duration by Type")
    st.write("This bar chart represents the **average duration** of different workout types.")
    avg_workout = df.groupby("Workout Type")["Workout Duration (mins)"].mean()
    st.bar_chart(avg_workout)

# Graph 2: Heart Rate vs Calories Burned
with col2:
    st.subheader("💓 Heart Rate vs Calories Burned")
    st.write("This scatter plot shows how **heart rate** correlates with **calories burned**.")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["Heart Rate (bpm)"], y=df["Calories Burned"], hue=df["Workout Type"], ax=ax, palette="coolwarm")
    st.pyplot(fig)

# Graph 3: Steps Taken Distribution
with col1:
    st.subheader("👟 Steps Taken Distribution")
    st.write("This histogram displays the **distribution of daily steps taken** by users.")
    fig, ax = plt.subplots()
    sns.histplot(df["Steps Taken"], bins=30, kde=True, ax=ax, color="#ff5733")
    st.pyplot(fig)

# Graph 4: Strength vs Cardio
with col2:
    st.subheader("🏋️ Strength vs Cardio")
    st.write("This line chart compares **calories burned** in **strength training vs cardio workouts**.")
    strength_vs_cardio = df.groupby("Workout Type")["Calories Burned"].mean()
    st.line_chart(strength_vs_cardio)

# Graph 5: Sleep Hours vs Water Intake
with col1:
    st.subheader("🌙 Sleep Hours vs Water Intake")
    st.write("This box plot compares **sleep duration** and **water intake levels**.")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["Sleep Hours"], y=df["Water Intake (liters)"], ax=ax, palette="Blues")
    ax.set_xticks(range(int(df["Sleep Hours"].min()), int(df["Sleep Hours"].max()) + 1, 1))
    ax.set_xticklabels(range(int(df["Sleep Hours"].min()), int(df["Sleep Hours"].max()) + 1, 1), rotation=45)
    st.pyplot(fig)

# Graph 6: Daily Calories Intake
with col2:
    st.subheader("🍏 Daily Calories Intake")
    st.write("This area chart shows the **daily calorie intake trends** of users.")
    st.area_chart(df["Daily Calories Intake"])

st.write("💡 *Stay consistent and track your progress to achieve your fitness goals!* 🏆")
