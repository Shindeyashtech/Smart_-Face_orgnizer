import os
import cv2
from deepface import DeepFace

# Set the base directory for your dataset
dataset_path = "dataset/"

def add_face(image_path, person_name):
    """
    Detects a face in the given image, saves it to the
    dataset, and creates a folder for the person if needed.
    """
    try:
        # Detect and extract faces using DeepFace's internal functions
        # This returns a list of dictionaries, one for each detected face
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",  # or 'mtcnn', 'retinaface', etc.
            enforce_detection=True,
            align=True,
            target_size=(160, 160)
        )

        if not faces:
            print(f"❌ Error: No face detected in {image_path}. Please try a different photo.")
            return

        # Use the first detected face
        face_img = faces[0]["face"]
        
        # Convert the face image from float to uint8 (required for saving)
        face_img = (face_img * 255).astype("uint8")

        # Create the person's directory if it doesn't exist
        person_folder = os.path.join(dataset_path, person_name)
        os.makedirs(person_folder, exist_ok=True)

        # Generate a unique filename and save the face image
        # Using a timestamp to avoid overwriting files
        import time
        filename = f"{person_name}_{int(time.time())}.jpg"
        save_path = os.path.join(person_folder, filename)
        
        cv2.imwrite(save_path, face_img)
        print(f"✅ Face detected and saved to {save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to use this script ---
# You need to provide the path to the image and the person's name.

# Example usage:
# new_image = "path/to/your/new_photo.jpg" 
# new_person = "rahul" # or a new person, e.g., "new_person"

# add_face(new_image, new_person)