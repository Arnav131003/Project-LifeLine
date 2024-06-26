import pandas as pd
import joblib

model = joblib.load('stroke_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

def predict_stroke():
    input_data = {}
    input_data['gender'] = input("Enter gender (Male/Female): ")
    input_data['age'] = float(input("Enter age: "))
    input_data['hypertension'] = int(input("Enter hypertension (0/1): "))
    input_data['heart_disease'] = int(input("Enter heart disease (0/1): "))
    input_data['ever_married'] = input("Enter marital status (Yes/No): ")
    input_data['work_type'] = input("Enter work type (Private/Self-employed/Govt_job/children/Never_worked): ")
    input_data['Residence_type'] = input("Enter residence type (Urban/Rural): ")
    input_data['avg_glucose_level'] = float(input("Enter average glucose level: "))
    input_data['bmi'] = float(input("Enter BMI: "))
    input_data['smoking_status'] = input("Enter smoking status (formerly smoked/never smoked/smokes/Unknown): ")

    input_data_encoded = {
        'age': input_data['age'],
        'hypertension': input_data['hypertension'],
        'heart_disease': input_data['heart_disease'],
        'avg_glucose_level': input_data['avg_glucose_level'],
        'bmi': input_data['bmi'],
        'gender_Male': 1 if input_data['gender'] == 'Male' else 0,
        'ever_married_Yes': 1 if input_data['ever_married'] == 'Yes' else 0,
        'work_type_Private': 1 if input_data['work_type'] == 'Private' else 0,
        'work_type_Self-employed': 1 if input_data['work_type'] == 'Self-employed' else 0,
        'work_type_children': 1 if input_data['work_type'] == 'children' else 0,
        'work_type_Govt_job': 1 if input_data['work_type'] == 'Govt_job' else 0,
        'Residence_type_Urban': 1 if input_data['Residence_type'] == 'Urban' else 0,
        'smoking_status_formerly smoked': 1 if input_data['smoking_status'] == 'formerly smoked' else 0,
        'smoking_status_never smoked': 1 if input_data['smoking_status'] == 'never smoked' else 0,
        'smoking_status_smokes': 1 if input_data['smoking_status'] == 'smokes' else 0,
    }

    input_df = pd.DataFrame([input_data_encoded])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return "May have Stroke" if prediction[0] == 1 else "No Stroke"

print("Prediction:", predict_stroke())
