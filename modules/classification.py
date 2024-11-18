from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib as plt
import seaborn as sns

class Animal:
    """
    Create animal instances
    """
    def __init__(self, 
                 hair=False, feathers=False, eggs=False, milk=False, predator=False,airborne = False, toothed=False, 
                 backbone=False, breathes=False, venomous=False, fins=False, legs=0, tail=False, aquatic = False,
                 domestic=False, catsize=False):
        self.hair = hair
        self.feathers = feathers
        self.eggs = eggs
        self.milk = milk
        self.airborne = airborne
        self.aquatic = aquatic
        self.predator = predator
        self.toothed = toothed
        self.backbone = backbone
        self.breathes = breathes
        self.venomous = venomous
        self.fins = fins
        self.legs = legs
        self.tail = tail
        self.domestic = domestic
        self.catsize = catsize
    def to_features(self):
        """
        Convert the animal's attributes into a feature array for classification.

        Returns:
            np.ndarray: A 2D array containing the animal's features as numerical values.
        """
        return np.array([[
            int(self.hair), int(self.feathers), int(self.eggs),
            int(self.milk), int(self.predator), int(self.airborne),
            int(self.toothed), int(self.backbone), int(self.breathes),
            int(self.venomous), int(self.fins), int(self.legs),
            int(self.tail), int(self.aquatic), int(self.domestic),
            int(self.catsize)]])

class AnimalClassifier:
    """
    A classifier for animals based on their features using a Decision Tree model.

    Attributes:
        model (DecisionTreeClassifier): The decision tree classifier.
    """
    def __init__(self):
        """
        Initialize the classifier with a DecisionTreeClassifier model.
        """
        self.model = DecisionTreeClassifier(random_state= 42)  # Your decision tree model will go here

    def train(self, X, y):
        """
        Train the classifier on the provided dataset.

        Args:
            X (np.ndarray): The feature matrix for training (n_samples, n_features).
            y (np.ndarray): The labels for training (n_samples,).
        """
        self.model.fit(X,y)

    def predict(self, animal):
        """
        Predict the class of a given animal based on its features.

        Args:
            animal (Animal): The animal instance to classify.

        Returns:
            str: The predicted class label.
        """
        features = animal.to_features()
        prediction = self.model.predict(features)
        return prediction[0]
    
    def matrix_and_scores(self, X_test, y_test):
        """
        Evaluate the classifier using the test dataset and generate performance metrics.

        Args:
            X_test (np.ndarray): The feature matrix for testing (n_samples, n_features).
            y_test (np.ndarray): The true labels for testing (n_samples,).

        Returns:
            tuple: A tuple containing:
                - accuracy (float): The accuracy of the classifier.
                - report (str): The classification report.
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)      
        report = classification_report(y_test, y_pred)      
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
        plt.title(f'Confusion Matrix for Classifier Tree')
        plt.xlabel('Predicted Category')
        plt.ylabel('Actual Category')
        plt.show()
        
        return accuracy, report




