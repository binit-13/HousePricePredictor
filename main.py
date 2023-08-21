# flask, scikit-learn, pandas, pickle-mixin
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('/Users/ruruthapa/Desktop/ML/machinelearning/HousePricePredictor/Housing_Cleaned_Data.csv')
pipe = pickle.load(open('/Users/ruruthapa/Desktop/ML/machinelearning/HousePricePredictor/linearModel.pkl', 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    BHK = request.form.get('BHK')
    bath = request.form.get('Bath')
    sqft = request.form.get('sqft')

    print(location, BHK, bath, sqft)
    input = pd.DataFrame([[location,sqft,bath,BHK]],columns=['location','total_sqft','bath','BHK'])
    prediction = np.round(pipe.predict(input)[0]*100000,2)
    prediction = "{:,}".format(prediction)
    print(prediction)
    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True)