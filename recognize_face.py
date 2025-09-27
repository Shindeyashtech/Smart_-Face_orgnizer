import pickle
from deepface import DeepFace
import cv2
import os

# Load the trained classifier
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Path to the image you want to recognize
new_image_path = "unseen_photo.jpg" 

# Check if the file exists
if not os.path.exists(new_image_path):
    print(f"Error: {new_image_path} not found.")
else:
    try:
        # Generate embedding for the new face
        embedding = DeepFace.represent(img_path=new_image_path, model_name="Facenet")[0]["embedding"]
        
        # Predict the person's name using the classifier
        predicted_name = classifier.predict([embedding])[0]
        
        # Get the confidence score
        confidence = max(classifier.predict_proba([embedding])[0]) * 100
        
        print(f"Predicted person: {predicted_name} with confidence: {confidence:.2f}%")

    except Exception as e:
        print(f"Error during recognition: {e}")