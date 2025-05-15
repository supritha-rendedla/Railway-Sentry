from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO  # Ensure correct import

app = Flask(__name__)

# Path for uploaded images and ensure 'uploads' folder exists
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the YOLO model
model = YOLO("yolo11n (4).pt")  # Replace with the correct path to your model

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_page')
def prediction_page():
    return render_template("prediction_page.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return redirect(url_for('prediction_page'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('prediction_page'))

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform object detection
        detected_image_path = detect_object(file_path)

        # Return relative path of the detected image for template rendering
        relative_image_path = f"uploads/{file.filename}"

        return render_template("prediction_page.html", detected_image=relative_image_path)

    return redirect(url_for('prediction_page'))

def detect_object(image_path):
    # Perform YOLOv5 or YOLOv8 detection
    results = model.predict(image_path, save=True, save_dir=app.config['UPLOAD_FOLDER'])
    
    # Assuming that `results` saves the output to the `save_dir`
    detected_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "detections.jpg")  # Adjust as needed
    return detected_image_path

if __name__ == "__main__":
    app.run(debug=True)
