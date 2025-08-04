import cv2
import numpy as np
import os
from datetime import datetime
import pickle
import shutil

class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_file = 'face_recognition_model.yml'
        self.label_file = 'face_labels.pkl'
        self.faces_dir = 'faces'
        self.min_face_size = (30, 30)
        self.confidence_threshold = 50
        self.load_or_create_model()

    def load_or_create_model(self):
        if os.path.exists(self.model_file) and os.path.exists(self.label_file):
            self.face_recognizer.read(self.model_file)
            with open(self.label_file, 'rb') as f:
                self.label_dict = pickle.load(f)
            print("Loaded existing face recognition model")
        else:
            self.label_dict = {}
            print("Created new face recognition model")

    def save_model(self):
        self.face_recognizer.write(self.model_file)
        with open(self.label_file, 'wb') as f:
            pickle.dump(self.label_dict, f)
        print("Model saved successfully")

    def create_face_database(self):
        cap = None
        try:
            if not os.path.exists(self.faces_dir):
                os.makedirs(self.faces_dir)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")

            name = input("Enter the person's name: ")
            face_count = 0
            total_faces = 10  # Increased number of samples
            
            print("\nCapturing faces... Please move your head slightly for different angles.")
            print("Press 'c' to capture manually or wait for automatic capture.")
            print("Press 'q' to quit capturing.\n")

            while face_count < total_faces:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame")

                # Create a copy for display
                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=self.min_face_size
                )

                largest_face = None
                largest_area = 0

                # Find the largest face in the frame
                for (x, y, w, h) in faces:
                    if w * h > largest_area:
                        largest_area = w * h
                        largest_face = (x, y, w, h)

                if largest_face is not None:
                    x, y, w, h = largest_face
                    # Draw rectangle around the largest face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add capturing indicator
                    cv2.putText(display_frame, f"Capturing: {face_count+1}/{total_faces}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Quality indicator
                    quality = self.get_image_quality(gray[y:y+h, x:x+w])
                    quality_color = (0, 255, 0) if quality > 0.5 else (0, 0, 255)
                    cv2.putText(display_frame, f"Quality: {quality:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

                # Display help text
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                          (10, display_frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Capture Faces', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                if largest_face is not None:
                    x, y, w, h = largest_face
                    if key == ord('c') or (quality > 0.5 and face_count < total_faces):
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        
                        # Save face with timestamp to avoid overwriting
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{name}_{face_count}_{timestamp}.jpg"
                        filepath = os.path.join(self.faces_dir, filename)
                        
                        cv2.imwrite(filepath, face_roi)
                        face_count += 1
                        print(f"Captured face {face_count}/{total_faces}")
                        
                        # Add small delay between captures
                        cv2.waitKey(500)

            print(f"\nCapturing completed. {face_count} faces captured.")
            self.train_model()

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

    def get_image_quality(self, face_roi):
        # Basic image quality assessment
        if face_roi is None or face_roi.size == 0:
            return 0.0
        
        # Calculate brightness
        brightness = np.mean(face_roi)
        # Calculate contrast
        contrast = np.std(face_roi)
        
        # Normalize values
        brightness_score = min(brightness / 128.0, 1.0)
        contrast_score = min(contrast / 128.0, 1.0)
        
        # Combine scores
        quality_score = (brightness_score + contrast_score) / 2.0
        return quality_score

    def train_model(self):
        try:
            faces = []
            labels = []
            current_id = len(self.label_dict)

            print("\nTraining model with captured faces...")
            
            # Load and process all face images
            for filename in os.listdir(self.faces_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    name = filename.split('_')[0]
                    
                    if name not in self.label_dict:
                        self.label_dict[name] = current_id
                        current_id += 1
                    
                    path = os.path.join(self.faces_dir, filename)
                    face_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    
                    if face_img is not None:
                        faces.append(face_img)
                        labels.append(self.label_dict[name])

            if len(faces) > 0:
                self.face_recognizer.train(faces, np.array(labels))
                self.save_model()
                print(f"Training completed with {len(faces)} face samples.")
            else:
                print("No face samples found for training.")

        except Exception as e:
            print(f"Error during training: {str(e)}")

    def recognize_faces(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")

            print("\nStarting face recognition...")
            print("Press 'q' to quit, 'c' to capture new faces")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame")

                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=self.min_face_size
                )

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))

                    try:
                        label, confidence = self.face_recognizer.predict(face_roi)
                        confidence_pct = max(0, min(100, 100 - confidence))
                        
                        # Get name and color based on confidence
                        if confidence_pct > self.confidence_threshold:
                            name = [k for k, v in self.label_dict.items() if v == label][0]
                            color = (0, 255, 0)  # Green for high confidence
                        else:
                            name = "Unknown"
                            color = (0, 0, 255)  # Red for low confidence

                        # Draw face rectangle and name
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        text = f"{name} ({confidence_pct:.1f}%)"
                        cv2.putText(display_frame, text, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    except Exception as e:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Unknown", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show instructions
                cv2.putText(display_frame, "Press 'q' to quit, 'c' to capture new faces",
                          (10, display_frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Face Recognition', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.create_face_database()

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

    def backup_database(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"backup_{timestamp}"
            
            if os.path.exists(self.faces_dir):
                shutil.copytree(self.faces_dir, backup_dir)
                if os.path.exists(self.model_file):
                    shutil.copy2(self.model_file, backup_dir)
                if os.path.exists(self.label_file):
                    shutil.copy2(self.label_file, backup_dir)
                print(f"Backup created successfully in {backup_dir}")
            else:
                print("No database to backup")
        except Exception as e:
            print(f"Backup failed: {str(e)}")

    def list_known_faces(self):
        try:
            if not self.label_dict:
                print("\nNo faces in database.")
                return
            
            print("\nKnown faces in database:")
            print("-" * 30)
            for name, label_id in self.label_dict.items():
                count = sum(1 for f in os.listdir(self.faces_dir) if f.startswith(name))
                print(f"Name: {name} (ID: {label_id}, Samples: {count})")
            print("-" * 30)
        except Exception as e:
            print(f"Error listing faces: {str(e)}")

def main():
    system = FaceRecognitionSystem()
    
    while True:
        try:
            print("\nFace Recognition System")
            print("1. Create/Update Face Database")
            print("2. Start Face Recognition")
            print("3. List Known Faces")
            print("4. Backup Database")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                system.create_face_database()
            elif choice == '2':
                system.recognize_faces()
            elif choice == '3':
                system.list_known_faces()
            elif choice == '4':
                system.backup_database()
            elif choice == '5':
                print("\nExiting...")
                break
            else:
                print("\nInvalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
