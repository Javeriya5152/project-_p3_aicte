# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# import time

# # Suppress warnings
# import warnings
# warnings.filterwarnings("ignore")

# # Streamlit UI Setup
# st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")
# st.title("🏋️ Personal Fitness Tracker")
# st.write("Track your fitness journey with personalized insights and predictions.")

# # Sidebar for user input
# st.sidebar.header("User Input Parameters")
# def user_input_features():
#     age = st.sidebar.slider("Age", 10, 100, 25)
#     bmi = st.sidebar.slider("BMI", 15, 40, 22)
#     duration = st.sidebar.slider("Workout Duration (minutes)", 5, 120, 30)
#     heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 85)
#     steps = st.sidebar.slider("Steps Taken", 1000, 20000, 8000)
#     body_temp = st.sidebar.slider("Body Temperature (°C)", 36, 42, 37)
#     gender = st.sidebar.radio("Gender", ("Male", "Female"))
#     fitness_goal = st.sidebar.selectbox("Fitness Goal", ["Weight Loss", "Muscle Gain", "Endurance", "General Fitness"])
    
#     gender_encoded = 1 if gender == "Male" else 0
#     goal_encoded = {"Weight Loss": 0, "Muscle Gain": 1, "Endurance": 2, "General Fitness": 3}[fitness_goal]
    
#     return pd.DataFrame({
#         "Gender": [gender_encoded], "Age": [age], "BMI": [bmi], "Duration": [duration],
#         "Heart_Rate": [heart_rate], "Steps": [steps], "Body_Temp": [body_temp], "Goal": [goal_encoded]
#     })

# df = user_input_features()
# st.write("### Your Input Parameters:", df)

# # Progress Bar
# progress_bar = st.progress(0)
# for i in range(100):
#     progress_bar.progress(i + 1)
#     time.sleep(0.01)

# # Load and preprocess dataset
# calories = pd.read_csv("C:/Users/javer/OneDrive/Desktop/snakegame/calories.csv")
# exercise = pd.read_csv("C:/Users/javer/OneDrive/Desktop/snakegame/exercise.csv")

# data = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
# data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)

# # Check if "Steps" column exists before selecting it
# columns_needed = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
# if "Steps" in data.columns:
#     columns_needed.append("Steps")

# data = data[columns_needed]

# # One-hot encoding for gender
# data = pd.get_dummies(data, columns=["Gender"], drop_first=True)

# # Split dataset
# X = data.drop("Calories", axis=1)
# y = data["Calories"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model = RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=42)
# model.fit(X_train_scaled, y_train)

# # Process user input
# input_df = df.copy()
# if "Gender" in input_df.columns:
#     input_df.rename(columns={"Gender": "Gender_male"}, inplace=True)
#     input_df["Gender_male"] = df["Gender"]
#     input_df.drop(columns=["Gender"], inplace=True)

# # Ensure same feature names for transformation
# missing_cols = set(X.columns) - set(input_df.columns)
# for col in missing_cols:
#     input_df[col] = 0  # Add missing columns with 0 value
# input_df = input_df[X.columns]  # Reorder to match training data

# # Scale input and predict
# input_scaled = scaler.transform(input_df)
# predicted_calories = model.predict(input_scaled)
# st.subheader("🔥 Predicted Calories Burned:")
# st.write(f"### {round(predicted_calories[0], 2)} kilocalories")

# # Visualization
# st.write("### 📊 Fitness Progress & Insights")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.histplot(data["Calories"], bins=30, kde=True, color="blue", alpha=0.6)
# ax.axvline(predicted_calories[0], color="red", linestyle="--", label="Your Prediction")
# ax.set_title("Calories Burned Distribution")
# ax.legend()
# st.pyplot(fig)

# # Recommendations
# st.write("### 🏆 Personalized Workout Suggestions")
# if df["Goal"].values[0] == 0:
#     st.write("✅ Try HIIT workouts and cardio-based exercises to maximize calorie burn.")
# elif df["Goal"].values[0] == 1:
#     st.write("✅ Focus on strength training with progressive overload to gain muscle.")
# elif df["Goal"].values[0] == 2:
#     st.write("✅ Engage in endurance training like running, cycling, or swimming.")
# else:
#     st.write("✅ Maintain a balanced fitness routine with flexibility and core workouts.")

# st.write("### 🔄 Compare with Others")
# cal_range = [predicted_calories[0] - 50, predicted_calories[0] + 50]
# similar_data = data[(data["Calories"] >= cal_range[0]) & (data["Calories"] <= cal_range[1])]
# st.write(similar_data.sample(min(5, len(similar_data))))

# st.write("---")
# st.write("💡 _Tip: Regularly tracking your workouts will help improve your performance!_")
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import smtplib
import time
import warnings
import os  # To check file existence

# Suppress warnings
warnings.filterwarnings("ignore")

# App title and description
st.title("🚴‍♀️ Personal Fitness Tracker")
st.write("""
Welcome to your personalized fitness tracker! Input your parameters, calculate BMI, set your fitness goal, and get personalized workout suggestions.
""")

# Sidebar - User inputs
st.sidebar.header("Set Your Parameters 🎯")

# Weight Input
weight = st.sidebar.number_input("Weight (kg): ", min_value=0.0, max_value=200.0, value=0.0, step=0.1)

# Height Input in Feet and Inches
height_feet = st.sidebar.number_input("Height (feet): ", min_value=0, max_value=8, value=0, step=1)
height_inches = st.sidebar.number_input("Additional Height (inches): ", min_value=0, max_value=11, value=0, step=1)

# Convert height to centimeters
height_cm = round((height_feet * 30.48) + (height_inches * 2.54), 2)
st.sidebar.write(f"**Height in centimeters: {height_cm} cm**")

# Other Inputs
age = st.sidebar.slider("Age: ", 0, 100, 0)
duration = st.sidebar.slider("Duration (min): ", 0, 60, 0)
heart_rate = st.sidebar.slider("Heart Rate: ", 0, 200, 0)
body_temp = st.sidebar.slider("Body Temperature (C): ", 0, 50, 0)
gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"), index=1)
gender = 1 if gender_button == "Male" else 0

# BMI Calculation
if height_cm > 0:  # Avoid division by zero
    bmi = round(weight / ((height_cm / 100) ** 2), 2)
else:
    bmi = 0
st.sidebar.write(f"**Calculated BMI: {bmi}**")

# Fitness Goal Selection
st.sidebar.header("Set Your Fitness Goal 💪")
fitness_goal = st.sidebar.radio(
    "What is your primary fitness goal?",
    ("Lose Weight", "Build Muscle", "Improve Endurance", "Maintain Fitness"),
    index=0
)

# Provide workout suggestions based on fitness goal
st.write("### 🏆 Personalized Workout Suggestions")
if fitness_goal == "Lose Weight":
    st.write("✅ Try HIIT workouts and cardio-based exercises to maximize calorie burn.")
elif fitness_goal == "Build Muscle":
    st.write("✅ Focus on strength training with progressive overload to gain muscle.")
elif fitness_goal == "Improve Endurance":
    st.write("✅ Engage in endurance training like running, cycling, or swimming.")
elif fitness_goal == "Maintain Fitness":
    st.write("✅ Maintain a balanced fitness routine with flexibility and core workouts.")

# Data preparation
data_model = {
    "Age": age,
    "BMI": bmi,
    "Duration": duration,
    "Heart_Rate": heart_rate,
    "Body_Temp": body_temp,
    "Gender_male": gender
}

user_data = pd.DataFrame(data_model, index=[0])

# Display user input parameters
st.write("---")
st.header("Your Parameters 📋")
st.write("Please review your input parameters below:")
st.write(user_data)

# History File Path
history_file = "user_history.csv"

# Check if History File Exists and Initialize if Missing
if not os.path.exists(history_file):
    headers = ["Date", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male", "Predicted_Calories", "Fitness_Goal"]
    pd.DataFrame(columns=headers).to_csv(history_file, index=False)

# Load history with error handling for malformed rows
try:
    history = pd.read_csv(history_file)
except pd.errors.ParserError:
    st.warning("History file contains malformed rows. Reinitializing the file.")
    headers = ["Date", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male", "Predicted_Calories", "Fitness_Goal"]
    history = pd.DataFrame(columns=headers)
    history.to_csv(history_file, index=False)

# Button to proceed to prediction
if st.button("Proceed to Prediction"):
    st.write("Analyzing your input...")

    # Add a progress bar for user feedback
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Simulated data loading and processing (replace with real dataset)
    exercise = pd.read_csv("C:/Users/javer/OneDrive/Desktop/snakegame/exercise.csv")
    calories = pd.read_csv("C:/Users/javer/OneDrive/Desktop/snakegame/calories.csv")
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    # Adding BMI feature
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)

    # Prepare training and test datasets
    train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    # Features and labels
    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)

    X_train = train_data.drop("Calories", axis=1)
    y_train = train_data["Calories"]

    # Model training
    model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=1)
    model.fit(X_train, y_train)

    # Align user input columns with training data
    user_data = user_data.reindex(columns=X_train.columns, fill_value=0)

    # Prediction
    prediction = model.predict(user_data)

    # Display prediction
    st.write("---")
    st.header("Predicted Calories Burned 🔥")
    st.write(f"**{round(prediction[0], 2)} kilocalories**")

    # Save to history
    user_data["Predicted_Calories"] = round(prediction[0], 2)
    user_data["Date"] = datetime.now().strftime("%Y-%m-%d")
    user_data["Fitness_Goal"] = fitness_goal  # Save fitness goal in history

    # Ensure data consistency before appending
    expected_columns = ["Date", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male", "Predicted_Calories", "Fitness_Goal"]
    user_data = user_data.reindex(columns=expected_columns)
    user_data.to_csv(history_file, mode="a", header=False, index=False)

    st.write("---")
    st.success("Your progress has been saved! ✅")

    # Weekly visualization
    st.write("---")
    st.header("Your Weekly Progress 📊")
    try:
        history = pd.read_csv(history_file)
        st.line_chart(data={
            "Calories Burned": history["Predicted_Calories"][-7:],  # Last 7 entries
            "Duration (min)": history["Duration"][-7:]
        })
    except Exception as e:
        st.warning(f"Could not load progress history: {e}")

    # Send Email if 7 entries are available
    if len(history) >= 7:
        def send_weekly_email():
            # Summarize weekly data
            last_7_days = history.tail(7)  # Get data for the last 7 entries
            avg_duration = last_7_days["Duration"].mean()
            avg_heart_rate = last_7_days["Heart_Rate"].mean()
            avg_calories = last_7_days["Predicted_Calories"].mean()

            # Email content
            email_content = f"""
            Subject: Your Weekly Fitness Summary

            Hello! Here's your fitness summary for this week:
            - Average Workout Duration: {avg_duration:.2f} minutes
            - Average Heart Rate: {avg_heart_rate:.2f} bpm
            - Average Calories Burned: {avg_calories:.2f} kcal

            Keep up the great work! 💪
            """

            # Sending the email
            sender_email = "youremail@gmail.com"
            receiver_email = "receiveremail@gmail.com"
            password = "yourpassword"

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, email_content)

        send_weekly_email()
        st.success("Weekly email sent successfully!")
