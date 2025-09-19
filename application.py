# src/application.py
import os
import sys
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from src.exceptions import CustomException
from src.logger import logging
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "wav", "mp3"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            try:
                # Run prediction pipeline
                custom_data = CustomData(file_path=file_path)
                features = custom_data.get_data_as_features()

                pipeline = PredictPipeline()
                pred = pipeline.predict(features)

                return render_template("index.html", prediction=pred[0], filename=filename)

            except Exception as e:
                raise CustomException(e, sys)

    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    logging.info("🚀 Starting Flask application...")
    app.run(host="0.0.0.0", port=5000, debug=True)
