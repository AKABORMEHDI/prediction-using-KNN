import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image


digits = load_digits()
x = digits.data
y = digits.target

# entraînement le modèle
model = KNeighborsClassifier(n_neighbors=2)
model.fit(x, y)

new_image_path = "51.png"
new_image = Image.open(new_image_path).convert("L")  # Convertir en niveaux de gris

new_image = new_image.resize((8, 8))

new_image_array = np.array(new_image)
inverted_image_array = 255 - new_image_array
plt.imshow(inverted_image_array, cmap='gray')
plt.title("Nouvelle image")
plt.show()

new_image_flattened = inverted_image_array.flatten().reshape(1, -1)

#la prédiction avec le modèle
prediction = model.predict(new_image_flattened)

# Afficher la prédiction
print("La prédiction pour la nouvelle image est :", prediction[0])