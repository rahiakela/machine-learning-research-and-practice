import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression

bc_dataset = datasets.load_breast_cancer()
X, y = bc_dataset.data, bc_dataset.target
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1234)

# let's classify
clf = LogisticRegression(lr=0.01)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# let's get accuracy
acc = clf.accuracy(y_pred, y_test)
print(f"Accuracy: {acc:.3f}")
