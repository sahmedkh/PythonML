# Importing The necessary packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Creating a class of my own K-means implementation
class MyKMeans:
    def __init__(self, n_clusters=3, max_iters=300):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Convert DataFrame to NumPy array
        X = X.values if isinstance(X, pd.DataFrame) else X

        # Initialize centroids randomly from data points
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Assign labels to each instance based on centroids
            self.labels = self._assign_labels(X)

            # Calculate new centroids based on the assigned labels
            new_centroids = self._calculate_centroids(X)
            
            # Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

    def _assign_labels(self, X):
        # Calculate distances between instances and centroids
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))

        # Assign labels to instances based on closest centroid
        return np.argmin(distances, axis=0)

    def _calculate_centroids(self, X):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.n_clusters):
            # Update centroids by taking the mean of instances assigned to each cluster
            new_centroids[i] = np.mean(X[self.labels == i], axis=0)
        return new_centroids


# Providing data from the csv file and seperating features
df = pd.read_csv("C:/Users/ahmed/OneDrive/Documents/Code/PythonML/assignment_3/creditcard.csv")
df = df.dropna()
df = df.iloc[:, 1:18]
df = MinMaxScaler().fit_transform(df)  

# Creating and fitting the model
kmeans = KMeans(n_clusters=3, random_state=0, n_init=100)
kmeans.fit(df)
centroids = kmeans.cluster_centers_

# Implementing the model using my own code
# def euclidean_distance(x, centroid):
#     return np.sqrt(np.sum((x - centroid) ** 2))

# def custom_k_means(data, k=3, max_iters=100):
#     centroids = data.sample(k)
#     for _ in range(max_iters):
#         clusters = {i: [] for i in range(k)}

#         for index, row in data.iterrows():
#             distances = [euclidean_distance(row.values, centroid.values) for _, centroid in centroids.iterrows()]
#             cluster_idx = distances.index(min(distances))
#             clusters[cluster_idx].append(row)

#         new_centroids = pd.DataFrame([np.mean(cluster, axis=0) for cluster in clusters.values()])

#         if centroids.equals(new_centroids):
#             break

#         centroids = new_centroids

#     return centroids

# custom_centroids = custom_k_means(df)
my_kmeans = MyKMeans(n_clusters=3, max_iters=300)
my_kmeans.fit(df)

# Compare Results
# print("\nScikit-learn cluster centers:")
# np.set_printoptions(suppress=True)
# print(centroids.round(3))
# print("\nCustom K-means cluster centers:")
# print(custom_centroids.round(3))

# Create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Plotting the resulting clusters
ax1.scatter(df[:, 0], df[:, 6])
ax1.set_title("Original Plotted Data")
#ax1.legend()

ax2.scatter(df[:, 0], df[:, 6], c=kmeans.labels_)
ax2.set_title("Scikit-learn Model")
#ax2.legend()

ax3.scatter(df[:, 0], df[:, 6], c=my_kmeans.labels)
ax3.set_title("My Own Model")
#ax3.legend()

plt.show()

# plt.figure(figsize=(12, 6))
# for centroid in custom_centroids.values:
#     plt.scatter(centroid, [0]*len(centroid), marker='*', s=300, c='black')

# for i, centroid in enumerate(custom_centroids.values):
#     plt.scatter(df.iloc[kmeans.labels_ == i].index, [0] * np.sum(kmeans.labels_ == i), label=f'Cluster {i}')

# plt.scatter(centroids[:, 0], [0]*len(centroids), marker='*', s=300, c='red')
# plt.scatter(df.index, [0]*len(df), c=kmeans.labels_, cmap='viridis', label='Sklearn K-Means Clusters')
# plt.title('Comparison of Custom K-Means and Sklearn K-Means Clustering')
# plt.xlabel('Sample Index')
# plt.legend()
# plt.show()