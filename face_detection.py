import dlib
import cv2

# Load the pre-trained face detector
face_detector = dlib.get_frontal_face_detector()

# Load the image and detect faces
image = dlib.load_rgb_image('images.jpg')
faces = face_detector(image)

# Iterate over the detected faces
for face in faces:
   # Process each face
   # Extract the bounding box coordinates of the face
   x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
   # Draw a rectangle around the face on the image
   cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

   # Display the image with detected faces
   cv2.imshow("Face Detection", image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
