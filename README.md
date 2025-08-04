# Facial Recognition System

This is a real-time facial recognition system using Python, OpenCV, and the face_recognition library.

## Setup

1. Make sure you have Python installed on your system
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install required packages:
   ```
   pip install opencv-python face_recognition numpy
   ```

## Usage

1. Create an 'images' folder in the project directory
2. Add photos of people you want to recognize to the 'images' folder
   - Name the files with the person's name (e.g., "john.jpg", "sarah.png")
   - Make sure each image contains only one clear face
3. Run the program:
   ```
   python facial_recognition.py
   ```
4. Press 'q' to quit the program

## Features

- Real-time face detection and recognition
- Support for multiple faces in the frame
- Easy addition of new faces through image files
- Visual display of recognized faces with names

## Requirements

- Python 3.6+
- OpenCV
- face_recognition
- numpy
