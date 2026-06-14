import numpy as np
import joblib

model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")

age = int(input("Age: "))
sex = int(input("Sex (1=Male, 0=Female): "))
cp = int(input("Chest Pain Type: "))
trestbps = int(input("Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar: "))
restecg = int(input("Rest ECG: "))
thalach = int(input("Max Heart Rate: "))
exang = int(input("Exercise Angina: "))

input_data = np.array([[age, sex, cp, trestbps,
                        chol, fbs, restecg,
                        thalach, exang]])

input_data = scaler.transform(input_data)

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")
