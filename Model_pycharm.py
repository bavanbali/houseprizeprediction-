import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Bengaluru_House_Data.csv')

# Clean the 'location' column by filling NaN values with 'Unknown'
data['location'] = data['location'].fillna('Unknown')

# Load the pre-trained model
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route("/")
def index():
    # Get sorted unique locations from the cleaned data
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route("/predict", methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    try:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)

        if bhk <= 0 or bath <= 0 or sqft <= 0:
            return render_template('index.html', locations=data['location'].unique(), error="Please enter positive values for all inputs.")
    except ValueError:
        return render_template('index.html', locations=data['location'].unique(), error="Please enter valid numeric values.")

    input_data = pd.DataFrame([[locations, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    # 🚀 Manipulate the result: always positive + double it
    prediction = abs(pipe.predict(input_data)[0] * 1e5) * 3.54

    # Format the prediction using the Indian numbering system
    formatted_prediction = indian_number_format(prediction)

    return render_template('index.html', prediction=formatted_prediction, locations=data['location'].unique())

def indian_number_format(num):
    # Ensure we handle both integer and float values
    if isinstance(num, float):
        num = round(num, 2)

    # Convert the number to string
    num_str = str(num)

    # Separate the integer and decimal parts
    if '.' in num_str:
        integer_part, decimal_part = num_str.split('.')
    else:
        integer_part = num_str
        decimal_part = '00'

    # Reverse the integer part for easier manipulation
    integer_part = integer_part[::-1]

    # Add commas after every two digits, except for the first three digits
    if len(integer_part) > 3:
        integer_part = integer_part[:3] + ',' + ','.join([integer_part[i:i+2] for i in range(3, len(integer_part), 2)])

    # Reverse back the integer part
    integer_part = integer_part[::-1]

    # Return formatted string with Indian numbering system
    return f"{integer_part}.{decimal_part[:2]}"  # Limiting to 2 decimal places

if __name__ == "__main__":
    app.run(debug=True, port=5001)
