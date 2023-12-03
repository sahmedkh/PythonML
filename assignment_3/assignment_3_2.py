# Importing The necessary packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Providing data from the csv file, selecting, and normalizing the features
df = pd.read_csv("C:/Users/ahmed/OneDrive/Documents/Code/PythonML/assignment_3/creditcard.csv")
df = df.iloc[:, 1:18]
df = df.dropna()
df = df[["BALANCE", "PURCHASES"]]  
df = MinMaxScaler().fit_transform(df)

# Creating and fitting the scikit-learn model
kmeans = KMeans(n_clusters=3, random_state=0, n_init=100)
kmeans.fit(df)
kmeans_labels = kmeans.labels_

# Creating and fitting our own model
def kmeans_clustering(X, n_clusters=2, max_iter=300):
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    for _ in range(max_iter):
        # Assign each data point to the closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        # Update centroids based on the mean of the assigned data points
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels

# Calling our model and getting the labels
cluster_labels = kmeans_clustering(df, n_clusters=3, max_iter=100)

# Plotting the resulting clusters
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plotting the resulting clusters
ax1.scatter(df[:, 0], df[:, 1])
ax1.set_xlabel('Balance')
ax1.set_ylabel('Purchases')
ax1.set_title("Original Plotted Data")

ax2.scatter(df[:, 0], df[:, 1], c=kmeans.labels_)
ax2.set_xlabel('Balance')
ax2.set_ylabel('Purchases')
ax2.set_title("Scikit-learn Model")

ax3.scatter(df[:, 0], df[:, 1], c=cluster_labels)
ax3.set_xlabel('Balance')
ax3.set_ylabel('Purchases')
ax3.set_title("My Own Model")

plt.show()