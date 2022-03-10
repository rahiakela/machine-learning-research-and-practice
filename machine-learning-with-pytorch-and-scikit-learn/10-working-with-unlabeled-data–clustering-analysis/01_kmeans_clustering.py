from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# simple two-dimensional dataset for the purpose of visualization
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  cluster_std=0.5,
                  centers=3,
                  shuffle=True,
                  random_state=0)

plt.scatter(X[:, 0], X[:, 1],
            c="white",
            marker="o",
            edgecolors="black",
            s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.grid()
plt.tight_layout()
plt.show()

# let’s apply it to our example dataset using the KMeans
km = KMeans(n_clusters=3,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# let’s now visualize the clusters that k-means identified in the dataset
# together with the cluster centroids
plt.scatter(X[y_km==0, 0], X[y_km==0, 1],
            s=50, c="lightgreen",
            marker="s",
            edgecolors="black",
            label="Cluster 1")
plt.scatter(X[y_km==1, 0], X[y_km==1, 1],
            s=50, c="orange",
            marker="o",
            edgecolors="black",
            label="Cluster 2")
plt.scatter(X[y_km==2, 0], X[y_km==2, 1],
            s=50, c="lightblue",
            marker="v",
            edgecolors="black",
            label="Cluster 3")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, c="red",
            marker="*",
            edgecolors="black",
            label="Centroids")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
