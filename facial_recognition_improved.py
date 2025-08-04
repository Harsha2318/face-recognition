import cv2
import numpy as np
import os
from datetime import datetime
import pickle

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize face detector with improved parameters
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize LBPH Face Recognizer with optimized parameters
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,          # Increased radius for better feature extraction
            neighbors=16,      # More neighbors for better pattern recognition
            grid_x=8,         # Finer grid for more detailed features
            grid_y=8,         # Finer grid for more detailed features
            threshold=45      # Lower threshold for stricter matching
        )
        
        self.model_file = 'face_recognition_model.yml'
        self.label_file = 'face_labels.pkl'
        self.faces_dir = 'faces'
        self.min_face_size = (60, 60)  # Increased minimum face size
        self.confidence_threshold = 45  # Adjusted confidence threshold
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

    def preprocess_face(self, face_roi):
        """Enhanced face preprocessing"""
        try:
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Apply histogram equalization for better contrast
            face_roi = cv2.equalizeHist(face_roi)
            
            # Apply Gaussian blur to reduce noise
            face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
            
            # Enhance edges
            face_roi = cv2.addWeighted(face_roi, 1.5, cv2.GaussianBlur(face_roi, (0, 0), 10), -0.5, 0)
            
            # Normalize pixel values
            face_roi = cv2.normalize(face_roi, None, 0, 255, cv2.NORM_MINMAX)
            
            return face_roi
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None

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
            total_faces = 15  # Increased number of samples
            
            print("\nCapturing faces... Please move your head slightly for different angles.")
            print("Follow these instructions for best results:")
            print("1. Look straight at the camera")
            print("2. Tilt head slightly left and right")
            print("3. Move head slightly up and down")
            print("4. Ensure good lighting")
            print("\nPress 'c' to capture manually or wait for automatic capture.")
            print("Press 'q' to quit capturing.\n")

            last_capture_time = datetime.now()
            min_capture_interval = 1.0  # Minimum interval between captures

            while face_count < total_faces:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame")

                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=self.min_face_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                largest_face = None
                largest_area = 0

                for (x, y, w, h) in faces:
                    if w * h > largest_area:
                        largest_area = w * h
                        largest_face = (x, y, w, h)

                if largest_face is not None:
                    x, y, w, h = largest_face
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Calculate quality metrics
                    quality = self.get_image_quality(face_roi)
                    brightness = np.mean(face_roi)
                    contrast = np.std(face_roi)
                    
                    # Display quality metrics
                    cv2.putText(display_frame, f"Quality: {quality:.2f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 0) if quality > 0.5 else (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Brightness: {brightness:.1f}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 0) if 100 < brightness < 200 else (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Contrast: {contrast:.1f}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 0) if contrast > 40 else (0, 0, 255), 2)

                    current_time = datetime.now()
                    time_diff = (current_time - last_capture_time).total_seconds()

                    if (quality > 0.6 and time_diff >= min_capture_interval) or \
                       cv2.waitKey(1) & 0xFF == ord('c'):
                        # Preprocess and save face
                        processed_face = self.preprocess_face(face_roi)
                        if processed_face is not None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{name}_{face_count}_{timestamp}.jpg"
                            filepath = os.path.join(self.faces_dir, filename)
                            
                            cv2.imwrite(filepath, processed_face)
                            face_count += 1
                            last_capture_time = current_time
                            print(f"Captured face {face_count}/{total_faces}")

                # Display progress
                cv2.putText(display_frame, f"Progress: {face_count}/{total_faces}", 
                          (10, display_frame.shape[0] - 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit",
                          (10, display_frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Capture Faces', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print(f"\nCapturing completed. {face_count} faces captured.")
            if face_count > 0:
                self.train_model()

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

    def get_image_quality(self, face_roi):
        """Enhanced image quality assessment"""
        if face_roi is None or face_roi.size == 0:
            return 0.0
        
        # Calculate various quality metrics
        brightness = np.mean(face_roi) / 255.0
        contrast = np.std(face_roi) / 255.0
        
        # Calculate blur metric using Laplacian
        blur = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        blur_score = min(blur / 500.0, 1.0)
        
        # Check face size
        size_score = min(face_roi.shape[0] * face_roi.shape[1] / (200 * 200), 1.0)
        
        # Combine scores with weights
        quality_score = (0.3 * brightness + 
                       0.3 * contrast + 
                       0.2 * blur_score + 
                       0.2 * size_score)
        
        return quality_score

    def train_model(self):
        try:
            faces = []
            labels = []
            current_id = len(self.label_dict)
            augmented_faces = []
            augmented_labels = []

            print("\nProcessing and augmenting face data...")
            
            for filename in os.listdir(self.faces_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    name = filename.split('_')[0]
                    
                    if name not in self.label_dict:
                        self.label_dict[name] = current_id
                        current_id += 1
                    
                    path = os.path.join(self.faces_dir, filename)
                    face_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    
                    if face_img is not None:
                        # Add original face
                        processed_face = self.preprocess_face(face_img)
                        if processed_face is not None:
                            faces.append(processed_face)
                            labels.append(self.label_dict[name])
                            
                            # Data augmentation
                            augmented = self.augment_face(processed_face)
                            augmented_faces.extend(augmented)
                            augmented_labels.extend([self.label_dict[name]] * len(augmented))

            # Combine original and augmented data
            all_faces = faces + augmented_faces
            all_labels = labels + augmented_labels

            if len(all_faces) > 0:
                print(f"Training model with {len(all_faces)} face samples...")
                self.face_recognizer.train(all_faces, np.array(all_labels))
                self.save_model()
                print("Training completed successfully!")
            else:
                print("No face samples found for training.")

        except Exception as e:
            print(f"Error during training: {str(e)}")

    def augment_face(self, face_img):
        """Data augmentation for better training"""
        augmented = []
        rows, cols = face_img.shape
        
        # Rotation
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(face_img, M, (cols, rows))
            augmented.append(rotated)
        
        # Brightness variation
        for alpha in [0.9, 1.1]:
            bright = cv2.convertScaleAbs(face_img, alpha=alpha, beta=0)
            augmented.append(bright)
        
        # Noise addition
        noise = np.random.normal(0, 5, face_img.shape).astype(np.uint8)
        noisy = cv2.add(face_img, noise)
        augmented.append(noisy)
        
        return augmented

    def recognize_faces(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")

            print("\nStarting face recognition...")
            print("Press 'q' to quit, 'c' to capture new faces")
            
            # Initialize face tracking
            tracking_faces = []
            max_track_age = 10  # frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to grab frame")

                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=self.min_face_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = self.preprocess_face(face_roi)
                    
                    if face_roi is not None:
                        try:
                            # Multiple predictions for stability
                            predictions = []
                            for _ in range(3):
                                label, confidence = self.face_recognizer.predict(face_roi)
                                predictions.append((label, confidence))
                            
                            # Average predictions
                            avg_label = max(set([p[0] for p in predictions]), 
                                         key=[p[0] for p in predictions].count)
                            avg_confidence = np.mean([p[1] for p in predictions 
                                                    if p[0] == avg_label])
                            
                            confidence_pct = max(0, min(100, 100 - avg_confidence))
                            
                            if confidence_pct > self.confidence_threshold:
                                name = [k for k, v in self.label_dict.items() 
                                      if v == avg_label][0]
                                color = (0, 255, 0)
                            else:
                                name = "Unknown"
                                color = (0, 0, 255)
                            
                            # Draw results
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                            text = f"{name} ({confidence_pct:.1f}%)"
                            cv2.putText(display_frame, text, (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                        except Exception as e:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), 
                                        (0, 0, 255), 2)
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

def main():
    system = FaceRecognitionSystem()
    
    while True:
        try:
            print("\nEnhanced Face Recognition System")
            print("1. Create/Update Face Database")
            print("2. Start Face Recognition")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                system.create_face_database()
            elif choice == '2':
                system.recognize_faces()
            elif choice == '3':
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
