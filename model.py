import numpy as np
import pandas as pd
import joblib

heart_disease_model = joblib.load('logistic_regression_model.pkl')
stroke_model = joblib.load('stroke_prediction_model.pkl')
stroke_scaler = joblib.load('scaler.pkl')
stroke_model_columns = joblib.load('model_columns.pkl')

def get_combined_input():
    age = int(input("Enter age: "))
    sex = int(input("Enter sex (0 = female, 1 = male): "))
    cp = int(input("Enter chest pain type (0, 1, 2, 3): "))
    trestbps = int(input("Enter resting blood pressure: "))
    chol = int(input("Enter serum cholesterol in mg/dl: "))
    fbs = int(input("Enter fasting blood sugar (1 if > 120 mg/dl, 0 otherwise): "))
    restecg = int(input("Enter resting electrocardiographic results (0, 1, 2): "))
    thalach = int(input("Enter maximum heart rate achieved: "))
    exang = int(input("Enter exercise induced angina (1 = yes, 0 = no): "))
    oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
    slope = int(input("Enter the slope of the peak exercise ST segment (0, 1, 2): "))
    ca = int(input("Enter number of major vessels (0-3) colored by fluoroscopy: "))
    thal = int(input("Enter thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect): "))
    hypertension = int(input("Enter hypertension (0/1): "))
    heart_disease = int(input("Enter heart disease (0/1): "))
    ever_married = input("Enter marital status (Yes/No): ")
    work_type = input("Enter work type (Private/Self-employed/Govt_job/children/Never_worked): ")
    residence_type = input("Enter residence type (Urban/Rural): ")
    avg_glucose_level = float(input("Enter average glucose level: "))
    bmi = float(input("Enter BMI: "))
    smoking_status = input("Enter smoking status (formerly smoked/never smoked/smokes/Unknown): ")
    
    heart_disease_input = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    stroke_input = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male': 1 if sex == 1 else 0,
        'ever_married_Yes': 1 if ever_married == 'Yes' else 0,
        'work_type_Private': 1 if work_type == 'Private' else 0,
        'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'work_type_children': 1 if work_type == 'children' else 0,
        'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
        'Residence_type_Urban': 1 if residence_type == 'Urban' else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
        'smoking_status_never smoked': 1 if smoking_status == 'never smoked' else 0,
        'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
    }

    return np.asarray(heart_disease_input).reshape(1, -1), pd.DataFrame([stroke_input]).reindex(columns=stroke_model_columns, fill_value=0)

heart_disease_input, stroke_input = get_combined_input()

heart_disease_prediction = heart_disease_model.predict(heart_disease_input)
stroke_input_scaled = stroke_scaler.transform(stroke_input)
stroke_prediction = stroke_model.predict(stroke_input_scaled)

if heart_disease_prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')

if stroke_prediction[0] == 1:
    print('The Person may have Stroke')
else:
    print('The Person does not have Stroke')
