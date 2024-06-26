import numpy as np
import joblib

# Load the saved model
model = joblib.load('logistic_regression_model.pkl')

# Function to convert user input to the required format
def get_user_input():
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

    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    return np.asarray(input_data).reshape(1, -1)

# Get user input
input_data_reshaped = get_user_input()

# Make a prediction
prediction = model.predict(input_data_reshaped)

# Print the prediction result
if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')
