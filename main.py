from flask import Flask, request, render_template, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from xgboost import XGBRegressor

app = Flask(__name__)

# Load dataset and train the model
df2 = pd.read_excel(r"fetch_california_housing.xlsx")

# Prepare data
X = df2.drop(['target'], axis=1)
Y = df2['target']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train model
model = XGBRegressor()
model.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Convert data to the correct format
    data = {key: [float(value)] for key, value in data.items()}
    input_df = pd.DataFrame.from_dict(data)

    # Make prediction
    prediction = model.predict(input_df)[0]
    print("The prediction is: ", prediction)
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)