from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "gender": request.form["gender"],
        "race/ethnicity": request.form["race_ethnicity"],
        "parental level of education": request.form["parental_education"],
        "lunch": request.form["lunch"],
        "test preparation course": request.form["test_preparation_course"]
    }

    # Encode inputs
    encoded_input = []
    for key, value in input_data.items():
        encoder = encoders[key]
        encoded_value = encoder.transform([value])[0]
        encoded_input.append(encoded_value)

    features = np.array([encoded_input])
    prediction = model.predict(features)
    result = "Pass" if prediction[0] == 1 else "Fail"

    return render_template("index.html", prediction=f"Student will {result}")

if __name__ == "__main__":
    app.run(debug=True)
