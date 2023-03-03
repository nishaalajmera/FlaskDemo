from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__,template_folder='template')

# Load model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
# Get the user input from the form
    glucose = float(request.form['glucose'])
    bp = float(request.form['bp'])
    bmi = float(request.form['bmi'])
    age = int(request.form['age'])

# Make a prediction using the trained model
    prediction = model.predict([[glucose, bp, bmi, age]])[0]

# Return the prediction result as JSON
    result = {
    'prediction': int(prediction),
    'message': 'You have high chances of having diabetes.' if prediction else 'You are healthy!'
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True)
