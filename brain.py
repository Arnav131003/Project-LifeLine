import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

file_path = '/Users/prathamagarwalla/Desktop/IMP/HACKATHONS/GE/ge-codes/healthcare-dataset-stroke-data.csv'
data = pd.read_csv(file_path)
data['bmi'].fillna(data['bmi'].mean(), inplace=True)
data = pd.get_dummies(data, drop_first=True)

X = data.drop('stroke', axis=1)
y = data['stroke']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

joblib.dump(model, 'stroke_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
