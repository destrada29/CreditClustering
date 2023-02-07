# Credit Customer Clustering

This code performs the clustering of credit customers using KMeans algorithm. The code performs the following steps:

1. Importing necessary libraries:

   - from sklearn.preprocessing import MinMaxScaler
   - from sklearn.decomposition import PCA
   - import numpy as np
   - import pandas as pd
   - import matplotlib.pyplot as plt
   - from sklearn.cluster import KMeans

2. Reading the credit customer data:

   - credito = pd.read_csv(r"CreditClustering\bdclientes2.csv", sep=';')

3. Exploratory Data Analysis (EDA):

   - Showing basic information of the data
   - Showing the first 5 records of the data
   - Descriptive statistics of the data

4. Data Preparation:

   - Dropping the 'Cedula' column
   - Creating dummy variables for 'Estado_civil' and 'Tipo de vivienda'
   - Dropping the 'Estado_civil' and 'Tipo de vivienda' columns
   - Concatenating the data with the dummy variables
   - Converting the data to a numpy array
   - Replacing 'M' with 1 and 'F' with 0 in the first column
   - Converting the numpy array back to a pandas dataframe
   - Scaling the data using MinMaxScaler

5. KMeans Clustering:

   - Plotting the sum of squared distances versus the number of clusters (to determine the optimal number of clusters)
   - Train the KMeans algorithm on the normalized credit data with 3 clusters
   - Store the cluster centers
   - Denormalize the cluster centers
   - Add the cluster labels as a column to the original credit data
   - Group the credit customers based on their cluster labels and calculate the mean value of each feature for each group

6. Results:
   - The final output is a table showing the mean value of each feature for each group of customers.

## License

This project is Open source
