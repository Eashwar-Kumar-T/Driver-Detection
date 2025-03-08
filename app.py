from flask import Flask, render_template, request, redirect, url_for, session
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
app = Flask(__name__)
app.secret_key = 'your_secret_key' 

model = load_model(r"C:\Users\vtraj\Desktop\CCP\model\driver_model.keras")  

activity_map =  {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}
                
# Ensure the uploads directory exists
UPLOAD_FOLDER = 'static/uploads'  # Ensure this is a static path
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    prediction = predict_activity(file_path)

    return redirect(url_for('result', prediction=prediction, image_file=file.filename))

@app.route('/result')
def result():
    prediction = request.args.get('prediction', '')
    image_file = request.args.get('image_file', '')
    return render_template('result.html', prediction=prediction, image_file=image_file)

def preprocess_image(image_path):
   
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (64, 64))

    img = img.reshape(-1, 64, 64, 1)  # Shape: (1, 64, 64, 1)
    
    return img

def predict_activity(image_path):
    img = preprocess_image(image_path)
    y_prediction = model.predict(img, verbose=0)
    predicted_activity = format(activity_map.get('c{}'.format(np.argmax(y_prediction))))
    return predicted_activity

@app.route('/upload_video')
def upload_video_page():
    return render_template('upload_video.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file uploaded", 400

    file = request.files['video']
    
    if file.filename == '':
        return "No selected file", 400

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    frames_and_predictions = predict_video(video_path)

    session['frames_and_predictions'] = frames_and_predictions

    return redirect(url_for('video_result', video_file=file.filename))

@app.route('/video_result')
def video_result():
    video_file = request.args.get('video_file', '')
    
    frames_and_predictions = session.get('frames_and_predictions', [])
    
    return render_template('video_result.html', video_file=video_file, frames_and_predictions=frames_and_predictions)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.reshape(1, 64, 64, 1)  # Shape: (1, 64, 64, 1)
    
    return frame

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  
    frames_and_predictions = []
    frame_interval = int(frame_rate * 10)  
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            preprocessed_frame = preprocess_frame(frame)
            y_prediction = model.predict(preprocessed_frame, verbose=0)
            predicted_activity = activity_map.get('c{}'.format(np.argmax(y_prediction)))

            frame_filename = f"frame_{frame_idx}.jpg"
            frame_path = os.path.join(UPLOAD_FOLDER, frame_filename)
            cv2.imwrite(frame_path, frame)

            frames_and_predictions.append((frame_filename, predicted_activity))

        frame_idx += 1

    cap.release()
    return frames_and_predictions

if __name__ == '__main__':
    app.run(debug=True)
