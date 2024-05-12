import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cv2
import os


def load_images(folder_path, label):
    images = []
    labels = []
    i=0
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (256, 256))
            images.append(image.flatten())
            labels.append(label)
            i=i+1
            if i==800:
                break
    return images, labels

cat_folder = "Dog_and_Cat/Cat_train"
cat_images, cat_labels = load_images(cat_folder, 0)

dog_folder = "Dog_and_Cat/Dog_train"
dog_images, dog_labels = load_images(dog_folder, 1)

X_train = np.concatenate((cat_images, dog_images), axis=0)
y_train = np.concatenate((cat_labels, dog_labels))

cat_folder = "Dog_and_Cat/Cat_test"
cat_images, cat_labels = load_images(cat_folder, 0)

dog_folder = "Dog_and_Cat/Dog_test"
dog_images, dog_labels = load_images(dog_folder, 1)

X_test = np.concatenate((cat_images, dog_images), axis=0)
y_test = np.concatenate((cat_labels, dog_labels))

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test))
y = y.reshape((-1, 1))
print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

#plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
#plt.show()
def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-np.clip(Z, -700, 700)))
    return A
def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)
def predict(X, W, b):
    A = model(X, W, b)
    return (A >= 0.5).astype(int)
def artificial_neuron(X, y, learning_rate = 0.01, n_iter = 1000):

    W, b = initialisation(X)

    Loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred))

    plt.plot(Loss)
    plt.show()

    return (W, b)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
W, b = artificial_neuron(X_normalized, y)

# Charger une nouvelle image à tester
test_image_path = "Dog_and_Cat/TEST/1.png"
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (256, 256))
test_image_flattened = test_image.flatten().reshape(1, -1)
# print(test_image_flattened.shape)
# Faire la prédiction avec le modèle
if not predict(test_image_flattened, W, b):
    ys = 'Cat'
else :
    ys = 'Dog'
print(ys)

