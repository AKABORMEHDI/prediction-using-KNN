import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import pandas as pd

dataset = pd.read_csv("digits_train_data.csv")

data = dataset.values 

X = data[:, 1:] 
y = data[:, 0]   

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

# Charger une nouvelle image depuis le PC
new_image_path = "7.png"  
new_image = Image.open(new_image_path).convert("L") 

new_image = new_image.resize((28, 28))

new_image_array = np.array(new_image)
inverted_image_array = 255 - new_image_array
plt.imshow(inverted_image_array, cmap='gray')
plt.title("Nouvelle image")
plt.show()

new_image_flattened = inverted_image_array.flatten().reshape(1, -1)


prediction = model.predict(new_image_flattened)

print("La prédiction pour la nouvelle image est :", prediction[0])