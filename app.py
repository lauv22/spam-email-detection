from flask import Flask, request, jsonify, render_template
from predict import predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data  = request.get_json()
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"error": "No email text provided."}), 400

    result = predict(email)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)