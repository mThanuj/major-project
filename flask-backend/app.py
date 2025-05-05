from flask import Flask, request, jsonify
from utils.extract_all_features import (
    extract_all_features,
)


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/extract-features", methods=["POST"])
def extract_features():
    if not request.data:
        return {"error": "No data provided"}, 400

    with open("/tmp/audio.wav", "wb") as f:
        f.write(request.data)

    features = extract_all_features("/tmp/audio.wav")

    return jsonify({"features": features})


app.run(debug=True)
