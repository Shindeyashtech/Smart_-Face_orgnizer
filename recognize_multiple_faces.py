import pickle
import os
import shutil
from deepface import DeepFace

# Set a minimum confidence threshold to add to the dataset
CONFIDENCE_THRESHOLD = 80.0

# Load the trained classifier
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

# Path to the folder containing unseen photos
unseen_photos_folder = "unseen_photos"
dataset_path = "dataset/"

# Loop through each file in the folder
for img_name in os.listdir(unseen_photos_folder):
    img_path = os.path.join(unseen_photos_folder, img_name)
    
    if not os.path.isfile(img_path):
        continue

    print(f"\nProcessing {img_name}...")

    try:
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]
        predicted_name = classifier.predict([embedding])[0]
        confidence = max(classifier.predict_proba([embedding])[0]) * 100
        
        print(f"✅ Predicted person: {predicted_name} with confidence: {confidence:.2f}%")

        if confidence >= CONFIDENCE_THRESHOLD:
            person_folder = os.path.join(dataset_path, predicted_name)
            
            # Check if the folder for this person already exists
            if not os.path.exists(person_folder):
                # Ask for user confirmation before creating a new folder
                user_input = input(f"Do you want to create a new folder for '{predicted_name}' and store this data? (y/n): ")
                
                if user_input.lower() != 'y':
                    print("Skipping folder creation and not adding photo.")
                    continue  # Skip to the next photo

            # Create the folder and move the photo
            os.makedirs(person_folder, exist_ok=True)
            shutil.move(img_path, os.path.join(person_folder, img_name))
            
            print(f"➡️ Moved {img_name} to {person_folder}")
        else:
            print(f"⚠️ Confidence too low ({confidence:.2f}%) to automatically add to dataset.")

    except Exception as e:
        print(f"❌ Could not process {img_name}: {e}")

print("\nAll photos processed.")