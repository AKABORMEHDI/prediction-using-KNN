from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        img = img.resize((100, 100))  # Resize image to a standard size
        img_array = np.array(img)
        images.append(img_array.flatten())  # Flatten image pixel values and add to list
    return images

# Paths to your image folders
cat_train_path = 'C:/Users/AM/Documents/IA/Dog_and_Cat/Cat_train'
cat_test_path = 'C:/Users/AM/Documents/IA/Dog_and_Cat/Cat_test'
dog_train_path = 'C:/Users/AM/Documents/IA/Dog_and_Cat/Dog_train'
dog_test_path = 'C:/Users/AM/Documents/IA/Dog_and_Cat/Dog_test'

# Load images for cats and dogs from training and testing folders
cat_train_images = load_images_from_folder(cat_train_path)
cat_test_images = load_images_from_folder(cat_test_path)
dog_train_images = load_images_from_folder(dog_train_path)
dog_test_images = load_images_from_folder(dog_test_path)

# Creating labels for the data
cat_train_labels = [0] * len(cat_train_images)
cat_test_labels = [0] * len(cat_test_images)
dog_train_labels = [1] * len(dog_train_images)
dog_test_labels = [1] * len(dog_test_images)

# Concatenate cat and dog data
X_train = cat_train_images + dog_train_images
X_test = cat_test_images + dog_test_images
y_train = cat_train_labels + dog_train_labels
y_test = cat_test_labels + dog_test_labels

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

#test
test_image_path = "t2.png"
test_image = Image.open(test_image_path)
test_image = test_image.resize((100, 100))  # Resize image to match training size
test_image_array = np.array(test_image).flatten().reshape(1, -1)

prediction = knn_classifier.predict(test_image_array)

if prediction[0] == 0:
    print("The image is a cat.")
else:
    print("The image is a dog.")