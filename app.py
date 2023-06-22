from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

app = Flask(__name__)\


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get feature values from form submission
    satisfaction_level = float(request.form['satisfaction_level'])
    last_evaluation = float(request.form['last_evaluation'])
    time_spend_company = int(request.form['time_spend_company'])
    work_accident = int(request.form['work_accident'])
    department_RandD = int(request.form['department_RandD'])
    department_hr = int(request.form['department_hr'])
    department_management = int(request.form['department_management'])
    salary_high = int(request.form['salary_high'])

    # Make prediction on new data
    new_data = [[satisfaction_level, last_evaluation, time_spend_company, work_accident, department_RandD, department_hr, department_management, salary_high]]
    prediction = model.predict(new_data)
    
    # Pass the prediction to the result page
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run()
