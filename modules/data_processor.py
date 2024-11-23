import pandas as pd

class DataProcessor:
    """
    Handles data loading and preprocessing tasks.
    """
    def __init__(self, zoo_path, class_path):
        self.zoo_path = zoo_path
        self.class_path = class_path
        self.zoo = None
        self.animal_class = None

    def load_data(self):
        """
        Loads data from CSV files.
        """
        self.zoo = pd.read_csv(self.zoo_path)
        self.animal_class = pd.read_csv(self.class_path)

    def preprocess_data(self):
        """
        Performs any preprocessing steps required.
        """
        # Example: Remove unnecessary columns, handle missing values, etc.
        pass

    def get_data(self):
        """
        Returns the loaded data.
        """
        return self.zoo, self.animal_class
