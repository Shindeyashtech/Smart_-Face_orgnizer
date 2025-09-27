from deepface import DeepFace
import os
import pickle

dataset_path = "dataset/"
embeddings_file = "embeddings.pkl"

# Store embeddings + labels
embeddings = []
labels = []

# Loop through dataset
for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue
    
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            # Extract embedding using DeepFace
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]
            embeddings.append(embedding)
            labels.append(person)
            print(f"Processed {img_name} for {person}")
        except Exception as e:
            print(f"Skipping {img_name}: {e}")

# Save embeddings + labels
with open(embeddings_file, "wb") as f:
    pickle.dump((embeddings, labels), f)

print("✅ Embeddings saved to", embeddings_file)
