import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Loading the Iris dataset
excel_file_path = 'iris.xls'
iris_data = pd.read_excel(excel_file_path)
print(iris_data)

# Create and train the RandomForestClassifier

X = iris_data.drop("Classification", axis=1)
y = iris_data["Classification"]
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling form submissions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        features = [
            float(request.form['sl']),
            float(request.form['sw']),
            float(request.form['pl']),
            float(request.form['pw'])
        ]

        # Make a prediction using the trained model
        prediction = model.predict([features])[0]

        # Return the prediction to the user
        if prediction:
            return render_template('result.html', prediction=prediction)
        else:
            return render_template('result.html', prediction="Prediction not available")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
