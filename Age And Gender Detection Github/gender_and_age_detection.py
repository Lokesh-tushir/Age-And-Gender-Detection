import cv2
import os
import numpy as np
from datetime import datetime

# Paths to the Caffe model and prototxt files for gender
gender_model_path = "C:\\Users\\Lokes\\Downloads\\gender_net.caffemodel"
gender_prototxt_path = "C:\\Users\\Lokes\\Downloads\\gender_deploy.prototxt"

# Paths to the Caffe model and prototxt files for age
age_model_path = "C:\\Users\\Lokes\\Downloads\\age_net.caffemodel"
age_prototxt_path = "C:\\Users\\Lokes\\Downloads\\age_deploy.prototxt"

if not all(os.path.exists(p) for p in [gender_model_path, gender_prototxt_path, age_model_path, age_prototxt_path]):
    raise FileNotFoundError("Model or prototxt file not found. Please check the file paths.")

# Initialize OpenCV's DNN module with the Caffe models
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt_path, gender_model_path)
age_net = cv2.dnn.readNetFromCaffe(age_prototxt_path, age_model_path)

# Define labels for gender and age classification
gender_labels = ['Male', 'Female']
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture with the video file path
video_path = "C:\\Users\\Lokes\\Downloads\\test4.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize counters for male and female
male_count = 0
female_count = 0

# Tracking variables
tracked_faces = []
max_distance = 50  # Maximum distance between faces in pixels

# Main loop to capture frames and make predictions
frame_skip = 2  # Process every 2nd frame
frame_count = 0

# Initialize video writer to save output video
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    new_tracked_faces = []

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Preprocess the face image for the models
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Gender Prediction
        gender_net.setInput(blob)
        gender_predictions = gender_net.forward()
        gender_confidence = gender_predictions[0]
        gender = gender_labels[gender_confidence.argmax()]
        gender_confidence_value = gender_confidence.max()

        # Age Prediction
        age_net.setInput(blob)
        age_predictions = age_net.forward()
        age = age_ranges[age_predictions[0].argmax()]

        # Check if the detected face is already tracked
        face_center = (x + w // 2, y + h // 2)
        new_face = True
        for tracked_face in tracked_faces:
            tracked_center = tracked_face["center"]
            distance = np.sqrt((face_center[0] - tracked_center[0]) ** 2 + (face_center[1] - tracked_center[1]) ** 2)
            if distance < max_distance:
                new_face = False
                break

        # If the face is new, count it
        if new_face:
            if gender == 'Female':
                female_count += 1
                border_color = (0, 255, 0)  # Green for female
            else:
                male_count += 1
                border_color = (255, 0, 0)  # Blue for male

            # Add the face to the tracked list
            tracked_faces.append({"center": face_center, "frames_left": 10})
        else:
            border_color = (0, 255, 255)  # Yellow for already tracked faces

        # Draw a rectangle around the face with the decision boundary color
        cv2.rectangle(frame, (x, y), (x+w, y+h), border_color, 2)  # Thickness set to 2

        # Put gender and age labels above the rectangle
        text = f"{gender}, {age}: {gender_confidence_value:.2f}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    # Update tracked faces
    tracked_faces = [
        {"center": face["center"], "frames_left": face["frames_left"] - 1}
        for face in tracked_faces
        if face["frames_left"] > 0
    ]

    # Display counts on the frame
    count_text = f"Males: {male_count}  Females: {female_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add current date and time to the frame
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Gender and Age Detection in Video', frame)

    # Initialize video writer after obtaining frame size
    if out is None:
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))

    # Write the frame to the output video
    out.write(frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
