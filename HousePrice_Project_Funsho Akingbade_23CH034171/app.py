
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model/house_price_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        features = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['TotalBsmtSF']),
            float(request.form['GarageCars']),
            float(request.form['FullBath']),
            float(request.form['YearBuilt'])
        ]
        prediction = round(model.predict([features])[0], 2)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
