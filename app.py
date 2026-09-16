import streamlit as st
import numpy as np
import joblib
import json

# ---- Load model, scaler, and feature order ----
# NOTE: these were saved with joblib.dump() in the notebook, so they must be
# loaded with joblib.load() -- pickle.load() on the scaler raises
# "UnpicklingError: STACK_GLOBAL requires str" because joblib's serialization
# format isn't a plain pickle stream.
model = joblib.load("AdaBoost_model.pkl")
scaler = joblib.load("scaler.pkl")

with open("selected_features.json") as f:
    selected_features = json.load(f)

st.title("Heart Disease Prediction")
st.write("Fill the following details to predict if a person has heart disease or not.")

# ---- Numeric inputs ----
age = st.number_input("Age", 20, 100, 50)
cholesterol = st.number_input("Cholesterol", 150, 349, 200)
blood_pressure = st.number_input("Blood Pressure", 90, 179, 120)
heart_rate = st.number_input("Heart Rate", 60, 99, 75)
blood_sugar = st.number_input("Blood Sugar", 70, 199, 100)
stress_level = st.number_input("Stress Level", 1, 10, 5)
exercise_hours = st.number_input("Exercise Hours (per week)", 0, 20, 4)

# ---- Chest Pain Type (one-hot: 4 columns) ----
chest_pain_type = st.selectbox(
    "Select Chest Pain Type",
    options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
)
chest_pain_typical = 1 if chest_pain_type == "Typical Angina" else 0
chest_pain_atypical = 1 if chest_pain_type == "Atypical Angina" else 0
chest_pain_nonanginal = 1 if chest_pain_type == "Non-anginal Pain" else 0
chest_pain_asymptomatic = 1 if chest_pain_type == "Asymptomatic" else 0

# ---- Smoking (one-hot: 3 columns) ----
smoking_choice = st.selectbox("Smoking", ["Current", "Former", "Never"])
smoking_current = 1 if smoking_choice == "Current" else 0
smoking_former = 1 if smoking_choice == "Former" else 0
smoking_never = 1 if smoking_choice == "Never" else 0

# ---- Alcohol Intake (one-hot: 3 columns) ----
alcohol_intake = st.selectbox("Alcohol Intake", ["Heavy", "Never", "Moderate"])
alcohol_intake_heavy = 1 if alcohol_intake == "Heavy" else 0
alcohol_intake_never = 1 if alcohol_intake == "Never" else 0
alcohol_intake_moderate = 1 if alcohol_intake == "Moderate" else 0

# ---- Exercise Induced Angina (one-hot: 2 columns) ----
exercise_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
exercise_angina_yes = 1 if exercise_angina == "Yes" else 0
exercise_angina_no = 1 if exercise_angina == "No" else 0

# ---- Diabetes (one-hot: 2 columns) ----
diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
diabetes_yes = 1 if diabetes == "Yes" else 0
diabetes_no = 1 if diabetes == "No" else 0

# ---- Family History (one-hot: 2 columns) ----
family_history = st.selectbox("Family History", options=["Yes", "No"])
family_history_yes = 1 if family_history == "Yes" else 0
family_history_no = 1 if family_history == "No" else 0

# ---- Predict ----
if st.button("Predict"):
    input_dict = {
        "Age": age,
        "Cholesterol": cholesterol,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Exercise Hours": exercise_hours,
        "Stress Level": stress_level,
        "Blood Sugar": blood_sugar,
        "Smoking_Current": smoking_current,
        "Smoking_Former": smoking_former,
        "Smoking_Never": smoking_never,
        "Alcohol Intake_Heavy": alcohol_intake_heavy,
        "Alcohol Intake_Moderate": alcohol_intake_moderate,
        "Alcohol Intake_Never": alcohol_intake_never,
        "Family History_No": family_history_no,
        "Family History_Yes": family_history_yes,
        "Diabetes_No": diabetes_no,
        "Diabetes_Yes": diabetes_yes,
        "Exercise Induced Angina_No": exercise_angina_no,
        "Exercise Induced Angina_Yes": exercise_angina_yes,
        "Chest Pain Type_Asymptomatic": chest_pain_asymptomatic,
        "Chest Pain Type_Atypical Angina": chest_pain_atypical,
        "Chest Pain Type_Non-anginal Pain": chest_pain_nonanginal,
        "Chest Pain Type_Typical Angina": chest_pain_typical,
    }

    # Build the row in the EXACT column order the model was trained on
    input_row = [input_dict[col] for col in selected_features]
    input_data = np.array([input_row], dtype=float)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ The model predicts this person **has** heart disease. "
                  f"(Predicted probability: {probability:.1%})")
    else:
        st.success(f"✅ The model predicts this person **does not have** heart disease. "
                    f"(Predicted probability of disease: {probability:.1%})")


