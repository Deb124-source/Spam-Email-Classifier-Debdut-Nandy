from flask import Flask, render_template, request

from train_model import get_model

flask_app = Flask(__name__)
flask_app.name = "Spamify"



# Load trained model directly
model, vectorizer = get_model()

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    transformed = vectorizer.transform([message])

    prediction = model.predict(transformed)[0]

    probabilities = model.predict_proba(transformed)[0]

    spam_probability = round(max(probabilities) * 100, 2)

    if prediction == "spam":

        result = "Spam Email 🚨"

    else:

        result = "Safe Email ✅"

    return render_template(
        "index.html",
        prediction=result,
        input_message=message
    )

if __name__ == "__main__":
    flask_app.run(debug=True)
