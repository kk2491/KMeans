from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np 

X = np.array([[5, 3],
			  [10, 15],
			  [15, 12],
			  [24, 10],
			  [30, 45],
			  [84, 70],
			  [71, 80],
			  [60, 78],
			  [55, 52],
			  [80, 91],])

plt.scatter(X[:, 0], X[:, 1], label = "True Position")
#plt.show()

kmeans_cluster = KMeans(n_clusters = 2)
kmeans_cluster = kmeans_cluster.fit(X)

print(kmeans_cluster)

print(kmeans_cluster.cluster_centers_)
print(kmeans_cluster.labels_)

plt.scatter(X[:,0], X[:,1], c = kmeans_cluster.labels_, cmap = "rainbow")
plt.show()


kmeans_cluster = KMeans(n_clusters = 3)
kmeans_cluster = kmeans_cluster.fit(X)
plt.scatter(X[:,0], X[:,1], c = kmeans_cluster.labels_, cmap = "rainbow")
plt.scatter(kmeans_cluster.cluster_centers_[:,0], kmeans_cluster.cluster_centers_[:,1], color = "black")
plt.show()


# Get the error 

error = []
for i in range(1, 11):
	kmeans_cluster = KMeans(n_clusters = 1)
	kmeans_cluster = kmeans_cluster.fit(X)
	error.append(kmeans_cluster.inertia_)

plt.plot(range(1, 11), error)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Error")
plt.show()

