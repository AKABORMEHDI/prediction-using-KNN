from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = "3.jpeg"
original_image = Image.open(image_path)

image_array = np.array(original_image)

inverted_image_array = 255 - image_array

inverted_image = Image.fromarray(inverted_image_array)


plt.figure(figsize=(10, 5))

#l'image originale
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Image originale")
plt.axis('off')

# l'image avec les couleurs inversées
plt.subplot(1, 2, 2)
plt.imshow(inverted_image, cmap='gray')
plt.title("Image avec couleurs inversées")
plt.axis('off')

plt.show()
