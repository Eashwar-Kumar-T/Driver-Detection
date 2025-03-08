#Driver Activity Detection

Overview

This project is a Flask-based web application that detects driver activities from images and videos using a deep learning model. It helps in identifying distracted driving behaviors, improving road safety.

Features

Image Upload & Prediction: Users can upload an image, and the model predicts the driver's activity.

Video Upload & Frame Analysis: Users can upload a video, and the model extracts frames at intervals, predicting driver activities for each frame.

Real-time Results: Predictions are displayed along with uploaded media.

User-friendly Interface: Built using Flask with HTML templates.

Technologies Used

Python

Flask

TensorFlow/Keras

OpenCV

NumPy

HTML/CSS

Dataset

The model is trained on a dataset of driver activities categorized as:

Safe driving

Texting (Left/Right)

Talking on the phone (Left/Right)

Operating the radio

Drinking

Reaching behind

Hair and makeup

Talking to a passenger

Installation & Setup

Prerequisites

Ensure you have Python 3.x installed.

Install Dependencies

pip install flask tensorflow opencv-python numpy

Running the Application

Clone this repository or download the project files.

Place your trained deep learning model (driver_model.keras) inside the model folder.

Run the Flask application:

python app.py

Open http://127.0.0.1:5000/ in a web browser.

Folder Structure

├── static/
│   ├── uploads/         # Stores uploaded images and video frames
├── templates/
│   ├── home.html        # Home Page
│   ├── index.html       # Image Upload Page
│   ├── result.html      # Image Prediction Result Page
│   ├── upload_video.html # Video Upload Page
│   ├── video_result.html # Video Prediction Result Page
├── model/
│   ├── driver_model.keras # Pre-trained model
├── app.py                # Flask application
├── README.md             # Project Documentation

How It Works

Image Processing

Users upload an image.

The image is preprocessed (resized, converted to grayscale).

The deep learning model predicts the driver's activity.

The result is displayed along with the uploaded image.

Video Processing

Users upload a video.

The system extracts frames at 10-second intervals.

Each frame is processed and analyzed.

Predictions are stored and displayed alongside the frames.

Deployment

For deployment on a cloud platform (e.g., AWS, Heroku):

Set up a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Run the app on a cloud server using Gunicorn or uWSGI.

Future Enhancements

Real-time webcam-based detection.

Mobile app integration.

Advanced deep learning models for higher accuracy.

License

This project is open-source and available for modification and distribution.
