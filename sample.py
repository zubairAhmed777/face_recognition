import dlib
import cv2
import numpy as np

# Load the pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Function to calculate Euclidean distance between two vectors
def euclidean_distance(face_descriptor1, face_descriptor2):
    return np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

# Method to compute the face descriptor for a known individual
def compute_face_descriptor(image_path):
    """
    Compute the 128-d face descriptor for an image.
    
    Args:
        image_path (str): Path to the image.
        
    Returns:
        list: 128-d face descriptor vector or None if no face is detected.
    """
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray_image)
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        return None
    
    # Compute face descriptor for the first detected face
    landmarks = predictor(gray_image, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
    
    return face_descriptor

# Load known face encodings and names
known_face_descriptors = []
known_names = []

# Example: Compute face descriptors for known images
known_images = [
    'images.jpg'
]
known_names_list = [
    'Robert Downey Jr'
]

for img_path, name in zip(known_images, known_names_list):
    descriptor = compute_face_descriptor(img_path)
    if descriptor:
        known_face_descriptors.append(descriptor)
        known_names.append(name)

# Load the image to recognize faces
image_path = 'images_1.jpg'  # Replace with your test image path
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray_image)
print(f"Number of faces detected: {len(faces)}")

# Loop through each detected face
for i, face in enumerate(faces):
    print(f"Processing face {i+1}")
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    # Get the landmarks/parts for the face
    landmarks = predictor(gray_image, face)
    face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)

    # Compare the detected face with known faces
    name = "Unknown"
    min_distance = float('inf')
    for known_descriptor, known_name in zip(known_face_descriptors, known_names):
        distance = euclidean_distance(face_descriptor, known_descriptor)
        if distance < 0.6 and distance < min_distance:  # Threshold to consider a match
            min_distance = distance
            name = known_name

    # Display the name below the face rectangle
    cv2.putText(image, name, (face.left(), face.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the image with detected faces and names
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

