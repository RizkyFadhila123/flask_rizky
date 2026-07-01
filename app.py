from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    model_names = ['Decision Tree', 'SVC']

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Pregnancies': int(request.form['Pregnancies']),
        'Glucose': int(request.form['Glucose']),
        'BloodPressure': int(request.form['BloodPressure']),
        'SkinThickness': int(request.form['SkinThickness']),
        'Insulin': int(request.form['Insulin']),
        'BMI': float(request.form['BMI']),
        'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
        'Age': int(request.form['Age'])
    }
    df = pd.DataFrame(data, index=[0])
    model_idx = int(request.form['model'][0])
    X = scaler.transform(df)
    model_idx = int(request.form['model'])
    y = model[model_idx].predict(X)
    prediction = 'Diabetic' if int(y[0]) == 1 else 'Non-Diabetic'
    return render_template('index.html', model_names=model_names, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)