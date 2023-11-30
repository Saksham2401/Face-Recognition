import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
from scipy.spatial.distance import cosine
import tensorflow as tf
import time
from keras_facenet import FaceNet

# Load the FaceNet model
facenet_model = FaceNet()

# Load the embeddings from the .npz file
data = np.load('ss-embd.npz')
embeddings_array = data['arr_0']
names = data['arr_1']

# Load the MTCNN face detection model
mtcnn_detector = MTCNN()

# Open the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or provide the path to a video file
# Initialize FPS variables
fps = 0
fps_start_time = time.time()
fps_frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if not ret:
        break

    # Detect faces in the frame using MTCNN
    faces = mtcnn_detector.detect_faces(frame)

    # Iterate over detected faces
    for face in faces:
        # Extract the face coordinates
        x, y, w, h = face['box']

        # Extract the face ROI from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face ROI
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_roi = cv2.resize(face_roi, (28, 28))
        face_roi = face_roi.astype('float32')

        # Convert the preprocessed face ROI to a 4D tensor
        face_tensor = np.expand_dims(face_roi, axis=0)

        # Generate embeddings using the FaceNet model
        face_embedding = facenet_model.embeddings(face_tensor)[0]

        # Compare the face embedding with stored embeddings
        distances = [cosine(face_embedding, emb) for emb in embeddings_array]

        # Check if distances list is not empty before finding the minimum distance
        if distances:
            min_distance = min(distances)
            min_distance_idx = np.argmin(distances)

            # Define a threshold for face recognition
            threshold = 0.7

            # Check if the minimum distance is below the threshold
            if min_distance < threshold:
                # Identify the recognized face
                recognized_face = embeddings_array[min_distance_idx]
                recognized_name = names[min_distance_idx]

                # Calculate the confidence level
                confidence = 100 - min_distance*100

                # Draw bounding box and label for the recognized face with confidence level
                label = f"{recognized_name} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Draw bounding box and label for unknown face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Calculate FPS
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        fps_frame_count = 0

    # Display the FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all
video_capture.release()
cv2.destroyAllWindows()
