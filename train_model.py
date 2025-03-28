import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load dataset
df = pd.read_csv("data/workout_fitness_tracker_data.csv")

# Encode categorical data (Gender and Workout Type)
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

le_workout = LabelEncoder()
df["Workout Type"] = le_workout.fit_transform(df["Workout Type"])

# Select features and target
X = df[["Age", "Gender", "Height (cm)", "Weight (kg)", "Workout Type", "Workout Duration (mins)"]]
y = df["Calories Burned"]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and preprocessing tools
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le_gender, "models/label_encoder_gender.pkl")
joblib.dump(le_workout, "models/label_encoder_workout.pkl")

print("✅ Model training complete. Files saved in 'models/'.")
