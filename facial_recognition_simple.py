import cv2
import numpy as np
import os
from datetime import datetime

def create_face_database():
    cap = None
    try:
        # Directory for storing face data
        if not os.path.exists('faces'):
            os.makedirs('faces')
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        face_count = 0
        name = input("Enter the name of the person: ")
        
        # Load the face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            print(f"Error: Cascade file not found at {cascade_path}")
            return
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            return
        
        while face_count < 5:  # Capture 5 face samples
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if face_count < 5:
                    face_img = frame[y:y+h, x:x+w]
                    face_filename = f'faces/{name}_{face_count}.jpg'
                    cv2.imwrite(face_filename, face_img)
                    face_count += 1
                    print(f"Captured face {face_count}/5")
            
            cv2.imshow('Capture Faces', frame)
            
            # Break if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nCapture stopped by user")
                break
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def recognize_faces():
    cap = None
    try:
        # Load face detector
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_detector.empty():
            print("Error: Could not load face detector")
            return

        # Initialize LBPH Face Recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Load training data
        faces = []
        labels = []
        label_dict = {}  # Dictionary to map label numbers to names
        current_label = 0

        # Check if faces directory exists and has files
        if not os.path.exists('faces') or not os.listdir('faces'):
            print("No face database found. Please capture some faces first.")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection', frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def main():
    while True:
        try:
            print("\nFace Recognition System")
            print("1. Create Face Database")
            print("2. Start Face Detection")
            print("3. Exit")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                create_face_database()
            elif choice == '2':
                recognize_faces()
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    finally:
        cv2.destroyAllWindows()
