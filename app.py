from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load model artifact
artifact = joblib.load("diabetes_pipeline.joblib")
model = artifact["model"]
features = artifact["features"]
threshold = artifact["metrics"]["threshold"]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # Collect input from form
            input_data = [float(request.form[feature]) for feature in features]
            x = pd.DataFrame([input_data], columns=features)

            # Get prediction
            proba = model.predict_proba(x)[:, 1][0]
            pred = int(proba >= threshold)

            prediction = "Diabetic" if pred == 1 else "Non-Diabetic"
            probability = round(proba * 100, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
