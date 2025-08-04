import cv2
print(f"OpenCV version: {cv2.__version__}")  # Print OpenCV version for debugging
import numpy as np
import os
from datetime import datetime
import sys
print(f"Python version: {sys.version}")  # Print Python version for debugging

def create_face_database():
    cap = None
    try:
        # Directory for storing face data
        if not os.path.exists('faces'):
            os.makedirs('faces')
        
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        face_count = 0
        name = input("Enter the name of the person: ")
        
        while face_count < 5:  # Capture 5 face samples
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    if face_count < 5:
                        face_img = frame[y:y+h, x:x+w]
                        face_filename = f'faces/{name}_{face_count}.jpg'
                        cv2.imwrite(face_filename, face_img)
                        face_count += 1
                        print(f"Captured face {face_count}/5")
                
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Capture Faces', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nCapture stopped by user")
                    break
                
            except KeyboardInterrupt:
                print("\nCapture interrupted by user")
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
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize LBPH Face Recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Check if faces directory exists
        faces_dir = 'faces'
        if not os.path.exists(faces_dir):
            print("No face database found. Please capture some faces first.")
            return
            
        # Load training data
        face_samples = []
        face_ids = []
        names = {}
        current_id = 0
        
        # Load face samples and create labels
        for filename in os.listdir(faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(faces_dir, filename)
                name = filename.split('_')[0]  # Get name from filename
                
                if name not in names:
                    names[name] = current_id
                    current_id += 1
                
                face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                face_samples.append(face_img)
                face_ids.append(names[name])
        
        # Train the recognizer
        if len(face_samples) > 0:
            recognizer.train(face_samples, np.array(face_ids))
            print("Training completed!")
            print("\nStarting face recognition... Press 'q' key to exit.")
        else:
            print("No face samples found!")
            return
        
        # Create reverse mapping of IDs to names
        id_to_name = {v: k for k, v in names.items()}
        
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
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                try:
                    # Predict the face
                    label, confidence = recognizer.predict(face_roi)
                    name = id_to_name.get(label, "Unknown")
                    
                    # Draw rectangle and name
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    confidence_text = f"{name} ({100 - confidence:.1f}%)"
                    cv2.putText(frame, confidence_text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                except:
                    # If recognition fails, just show the face box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # Add instruction text at the bottom of the frame
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nExiting face recognition...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def main():
    while True:
        try:
            print("\nFace Recognition System")
            print("1. Create Face Database")
            print("2. Start Face Recognition")
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