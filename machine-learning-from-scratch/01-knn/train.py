import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN


cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# let's plot the dataset
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors="k", s=20)
plt.show()

# let's classify it
clf = KNN(k=5)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print(f"Actual Labels[{len(y_test)}]: \n{y_test}")
print(f"Predicted Labels[{len(predictions)}]: \n{predictions}")

# let's calculate accuracy
acc = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {round(acc, 3):.3f}")




