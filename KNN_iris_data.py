import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors
import numpy as np

# Charger l'ensemble de données Iris
iris = datasets.load_iris()
x = iris.data[:, :2]  # Les deux premières caractéristiques
y = iris.target

# Créer le modèle k-NN et l'entraîner
clf = neighbors.KNeighborsClassifier(n_neighbors=15)
clf.fit(x, y)

# Créer un maillage pour le graphique
h = .02  # Taille de pas du maillage
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prédire la classe pour chaque point dans le maillage
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Mettre les résultats dans un graphique de dispersion coloré
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Afficher les points de données originaux
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title("Classification des classes Iris avec k-NN (k=15)")
plt.xlabel("Première caractéristique")
plt.ylabel("Deuxième caractéristique")
plt.show()
