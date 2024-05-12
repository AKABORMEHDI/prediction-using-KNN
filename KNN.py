import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split

# Charger l'ensemble de données Iris
iris = datasets.load_iris()
x = iris.data[:, :2]  # Les deux premières caractéristiques
y = iris.target

# Diviser l'ensemble de données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Créer les modèles k-NN et les entraîner sur l'ensemble d'entraînement
clf = neighbors.KNeighborsClassifier(n_neighbors=15)
clf1 = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
clf1.fit(x_train, y_train)

# Créer un maillage pour les graphiques
h = .02  # Taille de pas du maillage
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prédire la classe pour chaque point dans le maillage
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre les résultats dans des graphiques de dispersion colorés
Z = Z.reshape(xx.shape)
Z1 = Z1.reshape(xx.shape)

# Créer un sous-plot pour le premier graphique
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title("Classification avec k-NN (k=15)")
plt.xlabel("Première caractéristique")
plt.ylabel("Deuxième caractéristique")

# Créer un sous-plot pour le deuxième graphique
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z1, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title("Classification avec k-NN (k=5)")
plt.xlabel("Première caractéristique")
plt.ylabel("Deuxième caractéristique")

# Ajuster la mise en page pour éviter les chevauchements
plt.tight_layout()

# Afficher les graphiques
plt.show()
