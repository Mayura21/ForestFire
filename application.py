from flask import Flask, request, jsonify,  render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# Import regression and standard scalerpickle
regressor = pickle.load(open('models/regression.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', debug=True)


@app.route('/predict_data', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        temp = float(request.form['temperature'])
        rh = float(request.form['humidity'])
        ws = float(request.form['wind_speed'])
        rain = float(request.form['rain'])
        ffmc = float(request.form['ffmc'])
        dmc = float(request.form['dmc'])
        dc = float(request.form['dc'])
        isi = float(request.form['isi'])
        bui = float(request.form['bui'])
        region = float(request.form['region'])
        new_data_scaled = scaler.transform([[temp, rh, ws, rain, ffmc, dmc, isi, region]])
        result = regressor.predict(new_data_scaled)

        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')
    return


if __name__ == "__main__":
    app.run(host="0.0.0.0")
