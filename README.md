# US Crime Analysis

## Overview of Project

### Purpose
There are many crimes that happen in the United States. Some believe that bigger cities and more populated states have more crime rates. In this project, we will be analyzing a dataset describing 1973 violent crime rates by US State. The crimes considered are assault, murder, and rape. Also included is the percent of the population living in urban areas. Our goal is to use unsupervised machine learning methods such as Cluster Heat Maps, Principle Component Analysis, K-Means Clustering, Hierarchical Clustering, and DBSCAN to understand how violent crimes differ between states. 

### Data

The dataset is available as *USarrests.csv*. The dataset has 50 observations (corresponding to each state) on 4 variables: 
1. Murder: Murder arrests (per 100,000 residents)
2. Assault: Assault arrests (per 100,000 residents)
3. UrbanPop: Percent urban population
4. Rape: Rape arrests (per 100,000 residents)

You can read more about the dataset [here](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/USArrests.html). 

## Preliminary Exploratory Analysis

![us-crime-scatter](https://user-images.githubusercontent.com/29410712/179429045-7476b1c1-fe80-45de-92e9-4531dd56e933.png)

In our preliminary analysis, we can see that Murder and Assault is highly positively correlated with a correlation of 0.8. There are some correlation with Rape + Assault and Rape + Murder but not as strong. There is almost no correlation between the Urban Population with Murder and Assault.

### Analysis with Cluster Heat Map

![us-crime-heat](https://user-images.githubusercontent.com/29410712/179429129-f9b3fb27-773f-4f4b-b320-ffd95f49144c.png)

In our cluster heat map analysis, we can visualize that Urban Population has does not necessarily correlate with Murder and Assault. Which means that states with a higher population does not determine murders or assaults. There is some relationship between the number of Rapes, Murders, and Assaults. We can also see that Murder and Assault are similar within the states.

## Deeper Analysis

### Visualizing Data using Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is one of the most used unsupervised machine learning algorithms across a variety of applications: exploratory data analysis, dimensionality reduction, information compression, and data de-noising. PCA is a dimensionality reduction technique that transforms a set of features in a dataset into a smaller number of features called principal components while at the same time trying to retain as much information in the original dataset as possible. Since we have 4 different variables, we have a fourth dimensional data set. PCA can take 4 or more variables and make a two-dimensional PCA plot. PCA can also tell us which variable is the most valuable for clustering the data. It also can tell us how accurate the two-dimensional graph is. 

Principal Component Analysis calculates the average of each variable and using this average, finds the center of the data. It then shifts the data so that the center of the data is at the origin. From here, we input principal components. The principal components are vectors, but they are not chosen at random. The first principal component (PC1) is computed so that it explains the greatest amount of variance in the original features. Thus, it minimizes the distance between each data point on the graph (Sum of Squared) so PC1 is a linear combination of variables. It uses a scaled vector called the "Eigenvector" or "Singular Vector" for PC1. The sum of squared distances for the best fit line is the Eigenvalue for PC1. The second component (PC2) is orthogonal to the first, and it explains the greatest amount of variance left after the first principal component. Then we find PC3 and PC4 which are perpendicular to PC1 and PC2 that goes through the origin. The number of PCs is either the number of variables or the number of samples, whichever is smaller. 

Once all the principal components are figured out, you can use the eigenvalues to determine the proportion of variation that each PC accounts for. Then you can create a scree plot which is a graphical representation of the percentages of variation that each PC accounts for.

![pca](https://user-images.githubusercontent.com/29410712/179429289-cff10be1-9fed-4382-9dda-fd1a933c8f3c.png)

![scree-plot](https://user-images.githubusercontent.com/29410712/179429298-ec5602ad-dd69-4ee8-bedc-c08b68909080.png)

In this scree plot, we can see that PC1, PC2, and PC3 account for the vast majority of the variation. This means that a three-dimensional graph, using just PC1, PC2, and PC3 would be a good approximation of this four-dimensional graph since it would account for 95.66% of the variation in the data. Also, a two-dimensional graph would account for 86.76% of the variation in the data.

### K-Means Cluster Analysis

We will now cluster the states into four clusters using k-means. K-means cluster identify initial clusters and calculate the variances between each cluster or the Euclidean distance. It clusters all the remaining points, calculates the mean of each cluster and then reclusters based on the new means. It repeats until the clusters no longer change. It restarts the cluster until it finds the best overall cluster. It does as many reclustering as we tell it to do. It then comes back and returns to the optimal one.

![k-mean-cluster](https://user-images.githubusercontent.com/29410712/179429344-1f50a179-1cc5-4ba7-8a28-5eecb1b6b436.png)

| Clusters    | States |
| ----------- | -----------|
| 1   | Connecticut, Delaware, Hawaii, Indiana, Kansas, Massachusetts, New Jersey, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, Utah, Virginia, Washington, Wyoming |
| 2   | Alabama, Arkansas, Georgia, Louisiana, Mississippi, North Carolina, South Carolina, Tennessee |
| 3   | Alaska, Arizona, California, Colorado, Florida, Illinois, Maryland, Michigan, Missouri, Nevada, New Mexico, New York, Texas |
| 4   | Idaho, Iowa, Kentucky, Maine, Minnesota, Montana, Nebraska, New Hampshire, North Dakota, South Dakota, Vermont, West Virginia, Wisconsin |

![different-k-mean](https://user-images.githubusercontent.com/29410712/179429370-8d082b0d-4f10-47bf-8953-4f33f2a54cd2.png)

By using this we can determine the intra-cluster distance so that we can interpret the best k value.

![intra-cluster-distance](https://user-images.githubusercontent.com/29410712/179429395-d7ad781a-01ac-493c-8cbf-34d56bd2892a.png)

We can see that the total intra-cluster distance is large for $k = 1$ and decreases as we increase $k$, until $k=4$, after which it tapers off and gets only marginally smaller. The slope becomes constant after $k = 4$. This indicates that $k=4$ is a good choice.

![kmean-pca](https://user-images.githubusercontent.com/29410712/179429447-e53a4fcb-47c3-4abd-ba4d-a2207156af06.png)

Based on the updated PCA plot with the clustering, it is consistent with the clustering with the points split into four sections.

### Hierarchical Cluster Analysis

We will now use hierarchical clustering with complete linkage and Euclidean distance, cluster the states into four clusters. Then we will visualize the cluster results on top of the first two components.

Hierarchical clustering is often associated with heatmaps. It organizes the rows and columns based on similarity. This makes it easy to see correlations in the data.

![hierarchical-cluster](https://user-images.githubusercontent.com/29410712/179429472-2ebf40af-88b6-4f0e-9be7-b01f7e946e4f.png)

| Clusters    | States |
| ----------- | -----------|
| 1   | Alabama, Alaska, Georgia, Louisiana, Mississippi, North Carolina, South Carolina, Tennessee |
| 2   | Arkansas, Connecticut, Delaware, Hawaii, Indiana, Kansas, Kentucky, Massachusetts, Minnesota, Missouri, New Jersey, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, Utah, Virginia, Washington, Wisconsin, Wyoming |
| 3   | Arizona, California, Colorado, Florida, Illinois, Maryland, Michigan, Nevada, New Mexico, New York, Texas |
| 4   | Idaho, Iowa, Maine, Montana, Nebraska, New Hampshire, North Dakota, South Dakota, Vermont, West Virginia |

![hierarchical-cluster-pca](https://user-images.githubusercontent.com/29410712/179429715-5d38e3f6-3d55-480e-be84-0bec4bbf24c3.png)

The results are slightly different from the k-means. The data now is still split into four sections, but some of the states belong in different clusters.

### DBSCAN

Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996. It is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

![dbscan](https://user-images.githubusercontent.com/29410712/179429780-cd167133-32a9-4987-b9b0-2ff7eaf1098a.png)

| Clusters    | States |
| ----------- | -----------|
| 1   | Alabama, Georgia, Louisiana, Mississippi, South Carolina, Tennessee |
| 2   | Connecticut, Idaho, Indiana, Iowa, Kansas, Kentucky, Maine, Massachusetts, Minnesota, Missouri, Montana, Nebraska, New Hampshire, New Jersey, North Dakota, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, South Dakota, Utah, Vermont, Virginia, Washington, West Virginia, Wisconsin, Wyoming |
| 3   | Illinois, New York, Texas |
| 4   | Maryland, Michigan, New Mexico |

![dbscan-pca](https://user-images.githubusercontent.com/29410712/179429796-7dfa95b8-8554-44a6-8259-c27eee4dbeee.png)

These results on the PCA plot is a lot different than before. The DBSCAN is extremely sensitive to the changes in epsilon in the dataset.
