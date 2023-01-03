from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


X, y = datasets.make_regression(n_samples=100,
                                n_features=1,
                                noise=20,
                                random_state=4)

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1234)

# let's plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.show()

# let's train model
lin_reg = LinearRegression(lr=0.01)
lin_reg.fit(x_train, y_train)
predictions = lin_reg.predict(x_test)

# let's calculate error
mse = lin_reg.mse(y_test, predictions)
print(mse)

# let's plot prediction line
y_pred_line = lin_reg.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="b", linewidth=2, label="Prediction")
plt.show()

