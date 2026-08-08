
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {
            "student_id": [int(request.form["student_id"])],
            "gender": [request.form["gender"]],
            "study_time_hours": [float(request.form["study_time_hours"])],
            "attendance_percent": [float(request.form["attendance_percent"])],
            "sleep_hours": [float(request.form["sleep_hours"])],
            "parental_education": [request.form["parental_education"]],
            "internet_access": [request.form["internet_access"]],
            "extracurricular_activities": [
                request.form["extracurricular_activities"]
            ],
            "part_time_job": [request.form["part_time_job"]],
            "previous_grade": [float(request.form["previous_grade"])],
            "final_grade": [request.form["final_grade"]]
        }

        input_data = pd.DataFrame(data)

        transformed_data = pipeline.transform(input_data)

        prediction = model.predict(transformed_data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2) if prediction is not None else None
    )


if __name__ == "__main__":
    app.run(debug=True)