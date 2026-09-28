# Program to implement K-Means Clustering
# Dataset: Iris Dataset

# Step 1: Import required Python libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# Step 2: Load the Iris dataset
iris = load_iris()


# Step 3: Extract the input feature data
X = iris.data


# Step 4: Display the number of samples and features
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# Step 5: Display the names of the four features
print("\nFeature Names:")
for feature in iris.feature_names:
    print(feature)


# Step 6: Create the K-Means clustering model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# Step 7: Train the K-Means model
kmeans.fit(X)


# Step 8: Obtain the cluster labels
cluster_labels = kmeans.labels_


# Step 9: Display the cluster assignment of each sample
print("\nCluster Assignment of Each Sample:")
for i, label in enumerate(cluster_labels):
    print(f"Sample {i + 1}: Cluster {label}")


# Step 10: Display the cluster centers (centroids)
print("\nCluster Centers (Centroids):")
print(kmeans.cluster_centers_)


# Step 11: Count and display the number of samples
# assigned to each cluster
print("\nNumber of Samples in Each Cluster:")

for cluster in range(3):
    count = sum(cluster_labels == cluster)
    print(f"Cluster {cluster}: {count} samples")


# Step 12: Calculate and display K-Means inertia
print("\nK-Means Inertia:")
print(kmeans.inertia_)


# Step 13: Create a scatter plot using selected Iris features
# Using Petal Length and Petal Width
plt.figure(figsize=(10, 6))

plt.scatter(
    X[:, 2],
    X[:, 3],
    c=cluster_labels,
    cmap="viridis",
    s=50
)


# Step 14: Plot cluster centers using a different marker
plt.scatter(
    kmeans.cluster_centers_[:, 2],
    kmeans.cluster_centers_[:, 3],
    marker="X",
    s=200,
    c="red",
    edgecolor="black",
    label="Cluster Centers"
)


# Step 15: Add appropriate axis and title labels
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-Means Clustering of Iris Dataset")
plt.legend()


# Step 16: Display the final cluster visualization
plt.show()


# Step 17: Interpretation
print("\nInterpretation:")
print("The Iris dataset has been divided into three clusters")
print("using the K-Means clustering algorithm.")
print("Each cluster contains samples with similar feature values.")
print("The red X markers represent the cluster centroids.")
print("The inertia value represents the sum of squared distances")
print("between each sample and its assigned cluster center.")


# Final Result
print("\nResult:")
print("The Iris dataset is successfully divided into three clusters")
print("using the K-Means clustering algorithm.")
print("Cluster assignments, centroids, sample counts, inertia,")
print("and graphical visualization have been obtained.")
