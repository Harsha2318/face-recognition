import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path for the images directory
images_path = 'images'

# Lists to store face encodings and names
known_face_encodings = []
known_face_names = []

# Load sample images and learn how to recognize them
def load_known_faces():
    # List all files in the images directory
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        # Load the image
        face_image = face_recognition.load_image_file(os.path.join(images_path, image_file))
        
        # Get the face encoding
        face_encoding = face_recognition.face_encodings(face_image)[0]
        
        # Get the name from the filename (removing the extension)
        name = os.path.splitext(image_file)[0]
        
        # Add to our lists
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        
        print(f"Loaded face data for: {name}")

def process_frame(frame):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert from BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    
    # Check each face found in the frame
    for face_encoding in face_encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        face_names.append(name)
    
    # Draw results on frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since we scaled down the frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw label below face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
    
    return frame

def main():
    print("Loading known faces...")
    load_known_faces()
    
    print("Starting video capture...")
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open video capture device")
        return
    
    while True:
        # Get frame from video
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Display the frame
        cv2.imshow('Facial Recognition', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
