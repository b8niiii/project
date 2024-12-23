import pandas as pd
from modules.data_processor import DataProcessor
from modules.models import ClusteringModels
from modules.classification import Animal, AnimalClassifier

def load_and_prepare_data():
    """
    Loads and preprocesses the data.
    """
    data_processor = DataProcessor('animals_data/zoo.csv', 'animals_data/class.csv')
    data_processor.load_data()
    zoo, animal_class = data_processor.get_data()
    return zoo, animal_class

def perform_clustering(zoo_data):
    """
    Performs clustering algorithms and returns results.
    """
    cluster_models = ClusteringModels(zoo_data)
    # Perform PCA
    pca_data, explained_variance = cluster_models.perform_pca(n_components=14)
    # Perform KMeans clustering
    kmeans_labels = cluster_models.perform_kmeans(n_clusters=7)
    # Perform Agglomerative Clustering
    linkage_methods = ['ward', 'average', 'complete', 'single']
    agglomerative_labels = cluster_models.perform_agglomerative_clustering(
        n_clusters=7, linkage_methods=linkage_methods)
    return pca_data, kmeans_labels, agglomerative_labels, explained_variance

def compare_methods(agglomerative_labels, actual_labels):
    label_df = pd.DataFrame(agglomerative_labels)
    label_df["actual_label"] = actual_labels
    count = pd.DataFrame()
    for col in label_df.columns:
        count[col] = label_df[col].value_counts().sort_values(ascending= False).values
    return count, label_df
     
 


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


if __name__ == '__main__': # if the module is beign run directly
    # Load and prepare data
    zoo, animal_class = load_and_prepare_data()
    
    # Perform clustering
    pca_data, kmeans_labels, agglomerative_labels = perform_clustering(
        zoo.drop(columns=['animal_name', 'class_type']))
    
    # Train classifier
    classifier = train_classifier(zoo, animal_class)
    
