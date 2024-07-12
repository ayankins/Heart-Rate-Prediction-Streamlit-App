import streamlit as st
import pandas as pd
import joblib
import time

# Function to predict heart rate
def predict_heart_rate(loaded_model, new_data):
    model = loaded_model['model']
    model_features = loaded_model['features']

    # Encode categorical variables
    new_data_encoded = pd.get_dummies(new_data)

    # Ensure encoded dataframe has all model training columns
    for col in model_features:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0

    # Reorder columns to match model's training data
    new_data_encoded = new_data_encoded[model_features]

    # Simulate prediction delay
    time.sleep(2)  # Simulate processing time (adjust as needed)
    prediction = model.predict(new_data_encoded)
    return prediction[0]

# Main function to run the Streamlit app
def main():
    # UI customization
    st.set_page_config(page_title="Heart Rate Prediction App", page_icon=":heart:", layout="wide", initial_sidebar_state="expanded")

    st.title("Heart Rate Prediction App")

    # Load the model
    model_path = 'heart_rate_model_with_features.pkl'
    loaded_model = joblib.load(model_path)

    # Input form with sliders and dropdowns
    st.header("Input Your Information")
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Age (years)", min_value=0, max_value=100, value=25)
        Blood_Pressure = st.slider("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        Cholesterol = st.slider("Cholesterol (mg/dL)", min_value=100, max_value=300, value=200)
        Glucose = st.slider("Blood Glucose (mg/dL)", min_value=50, max_value=200, value=100)
    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age_Group = st.selectbox("Age Group", ["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "70+"])
        Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        Physical_Activity = st.selectbox("Regular Physical Activity", ["Yes", "No"])
        Smoking = st.selectbox("Smoking", ["Yes", "No"])
        Alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
        Family_History = st.selectbox("Family History of Heart Disease", ["Yes", "No"])

    # Prepare the input data for prediction
    new_data = {
        'Age': Age,
        'Gender': Gender,
        'Age_Group': Age_Group,
        'Blood_Pressure': Blood_Pressure,
        'Cholesterol': Cholesterol,
        'Diabetes': Diabetes,
        'Physical_Activity': Physical_Activity,
        'Smoking': Smoking,
        'Alcohol': Alcohol,
        'Family_History': Family_History,
        'Glucose': Glucose
    }
    new_data_df = pd.DataFrame([new_data])

    # Button to predict heart rate with heartbeat animation
    if st.button("Predict Heart Rate", key="predict_button"):
        # Display heartbeat animation while predicting
        st.image('heartbeat.svg', width=100)
        prediction = predict_heart_rate(loaded_model, new_data_df)
        st.success(f"Predicted Heart Rate: {prediction:.2f} bpm")

if __name__ == '__main__':
    main()
