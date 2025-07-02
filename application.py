from flask import Flask, request, jsonify,  render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application


@app.route("/")
def hello_world():
    return "<h1>Hello World!</h1>"


if __name__ == "__main__":
    app.run(host="0.0.0.0")