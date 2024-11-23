
# main.py

import pandas as pd
from modules.data_processor import DataProcessor
from modules.models import ClusteringModels
from modules.classification import Animal, AnimalClassifier

def load_and_prepare_data():
    """
    Loads and preprocesses the data.
    """
    data_processor = DataProcessor('data/zoo.csv', 'data/class.csv')
    data_processor.load_data()
    zoo, animal_class = data_processor.get_data()
    return zoo, animal_class

def perform_clustering(zoo_data):
    """
    Performs clustering algorithms and returns results.
    """
    cluster_models = ClusteringModels(zoo_data)
    # Perform PCA
    pca_data = cluster_models.perform_pca(n_components=8)
    # Perform KMeans clustering
    kmeans_labels = cluster_models.perform_kmeans(n_clusters=7)
    # Perform Agglomerative Clustering
    linkage_methods = ['ward', 'average', 'complete', 'single']
    agglomerative_labels = cluster_models.perform_agglomerative_clustering(
        n_clusters=7, linkage_methods=linkage_methods)
    return pca_data, kmeans_labels, agglomerative_labels

def train_classifier(zoo_data):
    """
    Trains the animal classifier.
    """
    classifier = AnimalClassifier()
    # Prepare features and labels
    X = zoo_data.drop(columns=['animal_name', 'class_type'])
    y = zoo_data['class_type']
    classifier.train(X, y)
    return classifier


"""I think we can get rid of these lines"""

if __name__ == '__main__': # if the module is beign run directly
    # Load and prepare data
    zoo, animal_class = load_and_prepare_data()
    
    # Perform clustering
    pca_data, kmeans_labels, agglomerative_labels = perform_clustering(
        zoo.drop(columns=['animal_name', 'class_type']))
    
    # Train classifier
    classifier = train_classifier(zoo, animal_class)
    
