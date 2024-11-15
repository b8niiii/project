from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
        #Convert animal attributes to a feature array for classification
        return np.array([[
            int(self.hair), int(self.feathers), int(self.eggs),
            int(self.milk), int(self.predator), int(self.airborne),
            int(self.toothed), int(self.backbone), int(self.breathes),
            int(self.venomous), int(self.fins), self.legs,
            int(self.tail), int(self.aquatic), int(self.domestic),
            int(self.catsize)]])

class AnimalClassifier:
    """
    Class to classify animals
    """
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state= 42)  # Your decision tree model will go here

    def train(self, X, y):
        self.model.fit(X,y)

    def predict(self, animal):
        features = animal.to_features()
        prediction = self.model.predict(features)
        return prediction[0]
    
    def matrix_and_scores(self, X_test, y_test):
        # Make predictions on the test set
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




