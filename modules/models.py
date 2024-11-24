from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import numpy as np

class ClusteringModels:
    """
    Handles clustering algorithms and PCA.
    """
    def __init__(self, data):
        self.data = data
        self.pca = None
        self.kmeans = None
        self.agglomerative_models = {} # to store distance methods as keys and clustering labels as values
        self.pca_data = None

    def perform_pca(self, n_components):
        """
        Performs PCA on the data and returns the transformed data along with residual variance.
        """
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(self.data)
        
        # Calculate the residual variance
        explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
       
        
        return self.pca_data, explained_variance


    def perform_kmeans(self, n_clusters):
        """
        Performs KMeans clustering.
        """
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(self.pca_data)
        return self.kmeans.labels_

    def perform_agglomerative_clustering(self, n_clusters, linkage_methods):
        """
        Performs Agglomerative Clustering with different linkage methods.
        """
        for method in linkage_methods:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
            model.fit(self.data)
            self.agglomerative_models[method] = model.labels_
        return self.agglomerative_models
