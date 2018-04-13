from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
X_moon, y_moon = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=0.5)

kmeans_moon = KMeans(2, random_state=0)
labels = kmeans_moon.fit_predict(X_moon)

model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
spectral_labels = model.fit_predict(X_moon)

plt.figure()
plt.subplot(2,1,1)
plt.scatter(X_moon[:,0], X_moon[:,1], c=labels, s=50, cmap='viridis')
centers = kmeans_moon.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=0.5)

plt.subplot(2,1,2)
plt.scatter(X_moon[:,0], X_moon[:,1], c=spectral_labels, s=50, cmap='viridis')

plt.show()
