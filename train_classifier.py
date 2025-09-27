import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load embeddings and labels
with open("embeddings.pkl", "rb") as f:
    embeddings, labels = pickle.load(f)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)

# Initialize and train the classifier (we'll use a Support Vector Machine as an example)
classifier = SVC(kernel="linear", probability=True)
classifier.fit(X_train, y_train)

# Optional: Evaluate the classifier on the test set
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classifier accuracy on test set: {accuracy * 100:.2f}%")

# Save the trained classifier
with open("classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

print("✅ Classifier trained and saved to classifier.pkl")