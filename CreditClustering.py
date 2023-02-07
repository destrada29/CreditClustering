# Importing necessary libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reading the credit customer data
credito = pd.read_csv(r"CreditClustering\bdclientes2.csv", sep=';')

# Showing basic information of the data
credito.info()

# Showing the first 5 records of the data
credito.head()

# Descriptive statistics of the data
credito.describe()

# Dropping the 'Cedula' column
credito_nuevo = credito.drop(['Cedula'], axis=1)

# Showing the first 5 records after the column drop
credito_nuevo.head()

# Creating dummy variables for 'Estado_civil' and 'Tipo de vivienda'
dummies = pd.get_dummies(credito_nuevo['Estado_civil'])
dummies_1 = pd.get_dummies(credito_nuevo['Tipo de vivienda'])

# Dropping the 'Estado_civil' and 'Tipo de vivienda' columns
credito_nuevo = credito_nuevo.drop(['Estado_civil'], axis=1)
credito_nuevo = credito_nuevo.drop(['Tipo de vivienda'], axis=1)

# Concatenating the data with the dummy variables
credito_nuevo = pd.concat([credito_nuevo, dummies], axis=1)
credito_nuevo = pd.concat([credito_nuevo, dummies_1], axis=1)

# Showing the first 5 records after the concatenation
credito_nuevo.head()

# Converting the data to a numpy array
credito_nuevo = np.array(credito_nuevo)

# Getting the shape of the numpy array
dimension = credito_nuevo.shape

# Replacing 'M' with 1 and 'F' with 0 in the first column
for i in range(dimension[0]):
    if credito_nuevo[i, 0] == 'M':
        credito_nuevo[i, 0] = 1
    else:
        credito_nuevo[i, 0] = 0

# Converting the numpy array back to a pandas dataframe
credito_nuevo = pd.DataFrame(credito_nuevo)

# Showing the first 5 records after the replacement
credito_nuevo.head()

# Scaling the data using MinMaxScaler
mm = MinMaxScaler()
credit_norm = mm.fit(credito_nuevo)
credit_norm = mm.transform(credito_nuevo)

# Converting the scaled data to a pandas dataframe
credit_norm = pd.DataFrame(credit_norm)

# Descriptive statistics of the scaled data
credit_norm.describe()


# Displaying the first five rows of the normalized credit data
credit_norm.head()

# List to store the sum of squared distances for each number of clusters
wcss = []

# Loop over the range 1 to 11 to try different numbers of clusters
for i in range(1, 11):
    # Initialize the KMeans algorithm with the specified number of clusters and maximum iterations
    kmeans = KMeans(n_clusters=i, max_iter=300)
    # Fit the KMeans algorithm to the normalized credit data
    kmeans.fit(credit_norm)
    # Append the sum of squared distances for this number of clusters to the list
    wcss.append(kmeans.inertia_)

# Plot the sum of squared distances versus the number of clusters
plt.plot(range(1, 11), wcss)
# Set the title and axis labels
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("wcss")

# Train the KMeans algorithm on the normalized credit data with 3 clusters
clustering = KMeans(n_clusters=3, max_iter=300)
clustering.fit(credit_norm)
# Store the cluster centers
clusters = pd.DataFrame(clustering.cluster_centers_)
# Display the cluster centers
print(clusters)

# Denormalize the cluster centers
cluster_desnor = mm.inverse_transform(clusters)
cluster_desnor = pd.DataFrame(cluster_desnor)
# Display the denormalized cluster centers
print(cluster_desnor)

# Add the cluster labels as a column to the original credit data
credito['KMeans_Clusters'] = clustering.labels_
# Display the first five rows of the credit data with the cluster labels
credito.head()

# Initialize PCA with 3 components
pca = PCA(n_components=3)
# Transform the normalized credit data using PCA
pca_credit = pca.fit_transform(credit_norm)
# Create a dataframe from the PCA-transformed data
pca_credit_df = pd.DataFrame(data=pca_credit, columns=[
                             'Componente1', 'Componente2', 'Componente3'])
# Add the cluster labels as a column to the PCA-transformed data
pca_add = pd.concat([pca_credit_df, credito['KMeans_Clusters']], axis=1)
# Display the first five rows of the PCA-transformed data with the cluster labels
pca_add.head()

# Create a 3D scatter plot of the first three principal components
grafica = plt.figure(figsize=(10, 10))
graf = grafica.add_subplot(projection="3d")
# Set the axis labels
graf.set_xlabel("componente 1")
graf.set_ylabel("Componente 2")
graf.set_zlabel("componente 3")
# Set the title
graf.set_title("Principal Components")

# Specify the colors for each cluster
colores = np.array(["blue", "green", "yellow"])

# Create a scatter plot using the first, second, and third components
# of the PCA transformed data (pca_add)
graf.scatter(xs=pca_add.Componente1, ys=pca_add.Componente2,
             zs=pca_add.Componente3, c=colores[pca_add.KMeans_Clusters], s=50)

# Display the scatter plot
plt.show()
