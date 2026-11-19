from machine_learning.kNN import KNN

X_train = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
y_train = [0, 0, 1, 2, 2, 1]

knn = KNN(k=3)
knn.fit(X_train, y_train)

X_new = [[2, 2], [8, 9]]
predicciones = knn.predict(X_new)
print("Predicciones:", predicciones)