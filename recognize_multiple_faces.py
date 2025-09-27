import pickle
from deepface import DeepFace
import os

# Load the trained classifier
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Path to the folder containing unseen photos
unseen_photos_folder = "unseen_photos"

# Loop through each file in the folder
for img_name in os.listdir(unseen_photos_folder):
    img_path = os.path.join(unseen_photos_folder, img_name)
    
    # Skip directories or non-image files
    if not os.path.isfile(img_path):
        continue

    print(f"\nProcessing {img_name}...")

    try:
        # Generate embedding for the new face
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]
        
        # Predict the person's name using the classifier
        predicted_name = classifier.predict([embedding])[0]
        
        # Get the confidence score
        confidence = max(classifier.predict_proba([embedding])[0]) * 100
        
        print(f"✅ Predicted person: {predicted_name} with confidence: {confidence:.2f}%")

    except Exception as e:
        print(f"❌ Could not process {img_name}: {e}")

print("\nAll photos processed.")