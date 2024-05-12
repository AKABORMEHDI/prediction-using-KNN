import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
df = pd.read_excel('C:/Users/akmehdi/Downloads/archaive/Date_Fruit_Datasets/Date_Fruit_Datasets.x1lsx)
df.head (10)
nom_fruit_cible = dict(zip(df.etiquette_fruit.unique(), df.nom_fruit.unique())) // dictionaire
print( nom_fruit_cible )
#valeurs caractéristiques et valeur cible
x = df [['poids', 'largeur', 'hauteur']]
y = df['nom_fruit']
#fractionner dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
#instanciation du modéle
modele_reglog = linear_model.LogisticRegression(random_state = ©,
solver = 'liblinear', multi_class = 'auto')
#training|
modele_reglog.fit(x_train,y_train)
#précision du modéle
precision = modele_reglog.score(x_test,y_test)
print(precision*100)
#prédiction
prediction_fruit = modele_reglog.predict([[160,5,8]])
print(predicltion_fruit)
#prédiction
prediction_fruit = modele_reglog.predict([[20,4.3,5.5]])
print(prediction_fruit)