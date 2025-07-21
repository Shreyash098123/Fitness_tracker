import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/workout_fitness_tracker_data.csv")
df.columns = df.columns.str.strip()

# Load trained model and preprocessors
try:
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_gender = joblib.load("models/label_encoder_gender.pkl")
    le_workout = joblib.load("models/label_encoder_workout.pkl")
except:
    model, scaler, le_gender, le_workout = None, None, None, None

# Streamlit UI
st.set_page_config(page_title="🏋️ Personal Fitness Tracker", layout="wide")

# Sidebar Inputs
st.sidebar.title("🎯 Enter Your Fitness Data")

age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", df["Gender"].unique())
height = st.sidebar.number_input("Height (cm)", min_value=120, max_value=220, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=150, value=70)
workout_type = st.sidebar.selectbox("Workout Type", df["Workout Type"].unique())
workout_duration = st.sidebar.number_input("Workout Duration (mins)", min_value=5, max_value=180, value=30)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=60, max_value=200, value=90)

gender_num = le_gender.transform([gender])[0] if le_gender else 0
workout_num = le_workout.transform([workout_type])[0] if le_workout else 0

input_data = np.array([[age, gender_num, height, weight, workout_num, workout_duration]])
input_data = scaler.transform(input_data) if scaler else input_data

if st.sidebar.button("🔥 Predict Calories Burned"):
    if model:
        prediction = model.predict(input_data)
        st.sidebar.success(f"🔥 Estimated Calories Burned: {round(prediction[0], 2)} kcal")
    else:
        st.sidebar.warning("⚠️ Model not available. Train the model first.")

st.title("🏋️‍♂️ Personal Fitness Tracker")
st.markdown("Monitor your fitness, track calories, and analyze workout trends!")

st.header("📚 General Insights")
st.markdown("""
**Who uses this tracker?**
- Most users are aged between **20 and 40** years old.
- There's an almost equal number of **male and female** users.
- Main goals include:
  - Managing weight
  - Increasing stamina
  - Monitoring hydration levels
  - Tracking heart rate and sleep

**Your Comparison to Other Users:**
- You are older than **{0:.1f}%** of users.
- You exercise longer than **{1:.1f}%** of users.
- Your heart rate during workouts is higher than **{2:.1f}%** of users.
""".format(
    np.mean(df["Age"] < age) * 100,
    np.mean(df["Workout Duration (mins)"] < workout_duration) * 100,
    np.mean(df["Heart Rate (bpm)"] < heart_rate) * 100 if "Heart Rate (bpm)" in df.columns else 0
))

st.header("🧠 Personalized Suggestions Based on Your Input")
suggestions = []
if workout_duration < df["Workout Duration (mins)"].mean():
    suggestions.append("Try increasing your workout duration gradually to reach the average of {:.1f} minutes.".format(df["Workout Duration (mins)"].mean()))
if heart_rate < 100:
    suggestions.append("Consider increasing workout intensity to boost cardiovascular efficiency.")
if weight > df["Weight (kg)"].mean():
    suggestions.append("Incorporate more cardio-based workouts and monitor calorie intake.")
if age > 50:
    suggestions.append("Include flexibility and low-impact strength exercises like yoga or pilates.")
if df[(df["Workout Type"] == workout_type) & (df["Workout Duration (mins)"] < workout_duration)].shape[0] < 5:
    suggestions.append("You're ahead of most users doing {}. Great work maintaining consistency!".format(workout_type))
if df["Daily Calories Intake"].mean() > 2500:
    suggestions.append("Daily calorie intake seems high on average. Monitor portions and focus on nutrient-dense food.")
if df["Sleep Hours"].mean() < 6.5:
    suggestions.append("Average user sleep is low. Aim for 7-8 hours for better recovery.")
if not suggestions:
    suggestions.append("Great job! Keep up your healthy lifestyle and continue tracking your progress.")

for tip in suggestions:
    st.success(f"💡 {tip}")

st.subheader("📊 Detailed Visual Analytics")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📌 Average Workout Duration by Type")
    fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
    df.groupby("Workout Type")["Workout Duration (mins)"].mean().plot(kind='bar', ax=ax_bar, color="skyblue")
    ax_bar.set_ylabel("Average Duration (mins)")
    ax_bar.set_xlabel("Workout Type")
    ax_bar.set_title("Workout Duration by Type")
    st.pyplot(fig_bar)

with col2:
    st.markdown("### 👟 Steps Taken Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
    sns.histplot(df["Steps Taken"], bins=20, kde=True, ax=ax_hist, color="#ff5733")
    ax_hist.set_xlabel("Steps Taken")
    ax_hist.set_ylabel("Number of Users")
    ax_hist.set_title("Steps Taken Distribution")
    st.pyplot(fig_hist)

col3, col4 = st.columns(2)
with col3:
    st.markdown("### 💓 Heart Rate vs Calories Burned")
    fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
    sns.scatterplot(x=df["Heart Rate (bpm)"], y=df["Calories Burned"], hue=df["Workout Type"], ax=ax_scatter, palette="coolwarm")
    ax_scatter.set_xlabel("Heart Rate (bpm)")
    ax_scatter.set_ylabel("Calories Burned")
    ax_scatter.set_title("Heart Rate vs Calories Burned")
    st.pyplot(fig_scatter)

with col4:
    st.markdown("### 🍽️ Daily Calories Intake")
    fig_cal, ax_cal = plt.subplots(figsize=(5, 4))
    sns.histplot(df["Daily Calories Intake"], bins=25, kde=True, color="green", ax=ax_cal)
    ax_cal.set_xlabel("Daily Calories Intake")
    ax_cal.set_ylabel("Number of Users")
    ax_cal.set_title("Daily Calories Intake Distribution")
    st.pyplot(fig_cal)

st.write("💡 *Stay consistent and track your progress to achieve your fitness goals!* 🌟")

with st.expander("📈 Compare Model Performance"):
    st.markdown("### 🔍 Evaluation of Different Regressors")

    df_eval = pd.read_csv("data/workout_fitness_tracker_data.csv")
    df_eval["Gender"] = LabelEncoder().fit_transform(df_eval["Gender"])
    df_eval["Workout Type"] = LabelEncoder().fit_transform(df_eval["Workout Type"])

    X = df_eval[["Age", "Gender", "Height (cm)", "Weight (kg)", "Workout Type", "Workout Duration (mins)"]]
    y = df_eval["Calories Burned"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR()
    }

    metrics = {
        "Model": [],
        "R² Score": [],
        "MAE": [],
        "RMSE": []
    }

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        metrics["Model"].append(name)
        metrics["R² Score"].append(r2_score(y_test, preds))
        metrics["MAE"].append(mean_absolute_error(y_test, preds))
        metrics["RMSE"].append(np.sqrt(mean_squared_error(y_test, preds)))

    metrics_df = pd.DataFrame(metrics).set_index("Model").round(4)

    # R2 Chart
    fig_r2, ax_r2 = plt.subplots()
    sns.barplot(x=metrics_df.index, y=metrics_df["R² Score"], ax=ax_r2, palette="Blues_d")
    ax_r2.set_title("Model R² Score Comparison")
    st.pyplot(fig_r2)

    # MAE & RMSE Charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### MAE Comparison")
        fig_mae, ax_mae = plt.subplots()
        sns.barplot(x=metrics_df.index, y=metrics_df["MAE"], ax=ax_mae, palette="Oranges")
        ax_mae.set_title("Mean Absolute Error")
        st.pyplot(fig_mae)

    with col2:
        st.markdown("#### RMSE Comparison")
        fig_rmse, ax_rmse = plt.subplots()
        sns.barplot(x=metrics_df.index, y=metrics_df["RMSE"], ax=ax_rmse, palette="Greens")
        ax_rmse.set_title("Root Mean Squared Error")
        st.pyplot(fig_rmse)

    # Show table
    st.markdown("### 📋 Full Metrics Table")
    st.dataframe(metrics_df)
