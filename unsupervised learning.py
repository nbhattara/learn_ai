# K-Means Clustering Example

from sklearn.cluster import KMeans
import numpy as np

# Data points
data = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 7],
    [9, 8],
    [8, 9]
])

# Model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

# Output
print("Cluster Labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)
