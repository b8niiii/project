from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import numpy as np

class ClusteringModels:
    """
    Handles clustering algorithms and PCA.
    """
    def __init__(self, data):
        """
        Initializes the ClusteringModels object with the dataset and placeholders 
        for PCA and clustering models.

        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            The dataset to be used for PCA and clustering.
        
        Attributes:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            The input dataset.
        pca : sklearn.decomposition.PCA or None
            Placeholder for the PCA model after initialization.
        kmeans : sklearn.cluster.KMeans or None
            Placeholder for the KMeans clustering model after initialization.
        agglomerative_models : dict
            A dictionary to store Agglomerative Clustering models, where keys are 
            linkage methods (e.g., 'ward', 'single', 'complete') and values are clustering labels.
        pca_data : numpy.ndarray or None
            The transformed dataset after applying PCA.
        """
        self.data = data
        self.pca = None
        self.kmeans = None
        self.agglomerative_models = {} # to store distance methods as keys and clustering labels as values
        self.pca_data = None

    def perform_pca(self, n_components):
         """
        Performs Principal Component Analysis (PCA) on the dataset.

        Parameters:
        -----------
        n_components : int
            The number of principal components to retain.

        Returns:
        --------
        tuple
            A tuple containing:
            - pca_data : numpy.ndarray
                The dataset transformed into the principal component space.
            - explained_variance : numpy.ndarray
                The cumulative explained variance for each principal component.
        """
         self.pca = PCA(n_components=n_components)
         self.pca_data = self.pca.fit_transform(self.data)
        
        # Calculate the residual variance
         explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
       
        
         return self.pca_data, explained_variance


    def perform_kmeans(self, n_clusters):
        """
        Performs KMeans clustering on the PCA-transformed data.

        Parameters:
        -----------
        n_clusters : int
            The number of clusters to form.

        Returns:
        --------
        numpy.ndarray
            An array of cluster labels for each data point.
        """
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(self.pca_data)
        return self.kmeans.labels_

    def perform_agglomerative_clustering(self, n_clusters, linkage_methods):
        """
        Performs Agglomerative Clustering with specified linkage methods.

        Parameters:
        -----------
        n_clusters : int
            The number of clusters to form.
        linkage_methods : list of str
            A list of linkage methods to use for clustering (e.g., 'ward', 'single', 'complete').

        Returns:
        --------
        dict
            A dictionary where keys are linkage methods and values are arrays of cluster labels.
        """
        for method in linkage_methods:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
            model.fit(self.data)
            self.agglomerative_models[method] = model.labels_
        return self.agglomerative_models
