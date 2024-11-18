import dlib
import csv
import cv2
import numpy as np
import pickle
from tqdm import tqdm

print('\n')
# Load the pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Method to compute the face descriptor for a known individual
def compute_face_descriptor(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        return None
    
    landmarks = predictor(gray_image, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
    
    return face_descriptor

# Load known image paths and names
'''known_images = [
    'images.jpg',
    'zubair.jpeg'
]
known_names = [
    'Robert Downey Jr.',
    'Zubair Ahmed'
]'''

def get_known_faces_names(source_file):
    known_images = list()
    known_names = list()
    with open(source_file, mode='r') as file:
        reader = csv.reader(file)
    
        # Skip the header row if there's one
        next(reader)
    
        # Iterate over each row in the CSV
        for row in reader:
            image_path, name = row
            known_images.append(image_path)
            known_names.append(name)
    return known_images, known_names

known_images, known_names = get_known_faces_names('known_faces.csv')

# Compute and store face descriptors
known_face_descriptors = []
for img_path, name in tqdm(zip(known_images, known_names), total=len(known_images), desc="Generating descriptors"):
    descriptor = compute_face_descriptor(img_path)
    if descriptor:
        known_face_descriptors.append((name, descriptor))

# Save the computed descriptors to a file using pickle
with open('known_face_descriptors.pkl', 'wb') as f:
    pickle.dump(known_face_descriptors, f)

print("\nFace descriptors saved successfully in 'known_face_descriptors.pkl'.")

