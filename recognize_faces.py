import dlib
import cv2
import numpy as np
import pickle
import sys
import os
sys.path.insert(0, './Silent-Face-Anti-Spoofing-master/src/')


#from test import test
from Silent_Face_Anti_Spoofing_master.test import test

# Load the pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Function to calculate Euclidean distance
def euclidean_distance(face_descriptor1, face_descriptor2):
    return np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

# Load the known face descriptors from the file
with open('known_face_descriptors.pkl', 'rb') as f:
    known_face_descriptors = pickle.load(f)

# Open the device camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray_frame)
        #print(f"Number of faces detected: {len(faces)}")

        # Loop through each detected face
        for i, face in enumerate(faces):
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            if test(image = frame,
                    #model_dir = os.path.join(os.path.dirname(__file__), 'Silent_Face_Anti_Spoofing_master', 'resources', 'anti_spoof_models')
                    model_dir = os.getcwd() + '\\Silent_Face_Anti_Spoofing_master\\resources\\anti_spoof_models\\',
                    device_id = 0
                    ) == 1:
                # Get the landmarks/parts for the face
                landmarks = predictor(gray_frame, face)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, landmarks)

                # Compare the detected face with known faces
                name = "Unknown"
                min_distance = float('inf')
                for known_name, known_descriptor in known_face_descriptors:
                    distance = euclidean_distance(face_descriptor, known_descriptor)
                    if distance < 0.6 and distance < min_distance:
                        min_distance = distance
                        name = known_name

            else:
                name = "Fake Image"
            # Display the name below the face rectangle
            cv2.putText(frame, name, (face.left(), face.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame with detected faces and names
        cv2.imshow("Face Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print('Done')
