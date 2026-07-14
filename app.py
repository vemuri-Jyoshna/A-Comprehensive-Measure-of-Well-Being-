from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('models/hdi_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    # Renders the introductory page
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    # Renders the input form page
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve input elements from form
        try:
            life_exp = float(request.form['life_exp'])
            exp_school = float(request.form['exp_school'])
            mean_school = float(request.form['mean_school'])
            gni = float(request.form['gni'])
            
            # Formulate the feature array for prediction
            features = np.array([[life_exp, exp_school, mean_school, gni]])
            
            # Predict
            prediction = model.predict(features)[0]
            # Rounding to 3 decimal places for readability
            output = round(prediction, 3)
            
            return render_template('indexnew.html', prediction_text=f'Predicted Human Development Index (HDI): {output}')
        except Exception as e:
            return render_template('indexnew.html', prediction_text=f'Error processing inputs: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)