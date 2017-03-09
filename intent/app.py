from flask import Flask, jsonify, request
from nlp import Embedder, Predictor


app = Flask(__name__)
embedder = Embedder()
predictor = Predictor(embedder, 'output/mymodel', 'output/intent_names.p')


@app.route("/predict", methods=['POST'])
def hello():
    data = request.get_json()
    text = data["text"]

    response = {}
    response["text"] = text
    response["prediction"] = predictor.predict(text)

    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
