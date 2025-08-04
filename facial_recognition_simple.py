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
        
        # Load face samples
        for filename in os.listdir('faces'):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join('faces', filename)
                name = filename.split('_')[0]  # Get name from filename
                
                # Assign numeric label to name
                if name not in label_dict:
                    label_dict[name] = current_label
                    current_label += 1
                
                # Read and preprocess face image
                face_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces.append(face_img)
                labels.append(label_dict[name])

        # Train the recognizer
        face_recognizer.train(faces, np.array(labels))
        print("Training completed!")

        # Create reverse mapping (label to name)
        name_dict = {label: name for name, label in label_dict.items()}

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

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Get face ROI
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))

                try:
                    # Predict the face
                    label, confidence = face_recognizer.predict(face_roi)
                    name = name_dict.get(label, "Unknown")
                    
                    # Calculate confidence percentage (100% - confidence, as lower confidence means better match)
                    confidence_pct = max(0, min(100, 100 - confidence))
                    
                    # Draw rectangle around face
                    color = (0, 255, 0) if confidence_pct > 50 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Show name and confidence
                    text = f"{name} ({confidence_pct:.1f}%)"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                except:
                    # If recognition fails
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Show quit instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition', frame)
            
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
