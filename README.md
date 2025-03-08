# Driver Behavior Detection

## Overview
This project is a **Flask-based web application** that detects driver activities from images and videos using a **deep learning model**. It helps in identifying distracted driving behaviors, improving road safety.

## Features
- **Image Upload & Prediction**: Users can upload an image, and the model predicts the driver's activity.
- **Video Upload & Frame Analysis**: Users can upload a video, and the model extracts frames at intervals, predicting driver activities for each frame.
- **Real-time Results**: Predictions are displayed along with uploaded media.
- **User-friendly Interface**: Built using Flask with HTML templates.

## Technologies Used
- **Python**
- **Flask**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy**
- **HTML/CSS**

## Dataset
The model is trained on a dataset of driver activities categorized as:
- Safe driving
- Texting (Left/Right)
- Talking on the phone (Left/Right)
- Operating the radio
- Drinking
- Reaching behind
- Hair and makeup
- Talking to a passenger

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.x** installed.

### Install Dependencies
```sh
pip install flask tensorflow opencv-python numpy
```

### Running the Application
1. Clone this repository or download the project files.
2. Place your trained deep learning model (`driver_model.keras`) inside the `model` folder.
3. Run the Flask application:
   ```sh
   python app.py
   ```
4. Open `http://127.0.0.1:5000/` in a web browser.

## Folder Structure
```
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
```

## How It Works
### Image Processing
1. Users upload an image.
2. The image is preprocessed (resized, converted to grayscale).
3. The deep learning model predicts the driver's activity.
4. The result is displayed along with the uploaded image.

### Video Processing
1. Users upload a video.
2. The system extracts frames at 10-second intervals.
3. Each frame is processed and analyzed.
4. Predictions are stored and displayed alongside the frames.

## Deployment
For deployment on a cloud platform (e.g., AWS, Heroku):
1. **Set up a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the app on a cloud server** using **Gunicorn or uWSGI**.

## Future Enhancements
- Real-time webcam-based detection.
- Mobile app integration.
- Advanced deep learning models for higher accuracy.

## License
This project is open-source and available for modification and distribution.
