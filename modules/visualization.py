import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
import streamlit as st

class Visualization:
    """
    Contains methods for creating various plots.
    """

    @staticmethod
    def plot_boxplot(data, column, title):
        """
        Creates a horizontal boxplot for a specified column.

        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset containing the column to plot.
        column : str
            The name of the column to visualize.
        title : str
            The title of the boxplot.

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the boxplot.
        """
        fig, ax = plt.subplots()
        data.boxplot(column=column, grid=False, vert=False, patch_artist=True, ax=ax)
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_pie_chart(values, labels, title):
        """
        Generates a pie chart with given values and labels.

        Parameters:
        -----------
        values : list or array-like
            The values for the pie chart.
        labels : list
            The labels for each segment.
        title : str
            The title of the pie chart.

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the pie chart.
        """
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.0f%%', colors = ['#7EC8E3', '#5B92E5', '#3C69E7', '#2052B2','#1B3A93','#12275E'])
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_confusion_matrix(cm, title):
        """
        Displays a confusion matrix as a heatmap.

        Parameters:
        -----------
        cm : array-like
            The confusion matrix to display.
        title : str
            The title of the heatmap.

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the heatmap.
        """
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        return fig

    @staticmethod
    def plot_dendrogram(method, df, title, cluster_threshold=None):
        """
        Creates a hierarchical clustering dendrogram.

        Parameters:
        -----------
        method : str
            The linkage method to use (e.g., 'single', 'complete', 'average').
        df : array-like
            The dataset.
        title : str
            The title of the dendrogram.
        cluster_threshold : int, optional
            The number of clusters to visualize by cutting the dendrogram.

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the dendrogram.
        """
        link = hierarchy.linkage(df, method = method)
        fig, ax = plt.subplots()
        hierarchy.dendrogram(link, ax=ax)
        if cluster_threshold:
            sorted(link[:, 2], reverse = True)[cluster_threshold] #sixth largest linkage distance = 7th cluster
        ax.set_title(title)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Distance')
        ax.set_xticks([])
        return fig

    @staticmethod
    def plot_elbow_graph(residual_variance):
        """
        Plots an elbow graph of residual variance to determine the optimal number of components.

        Parameters:
        -----------
        residual_variance : list or array-like
            Residual variance values for each number of components.

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the elbow graph.
        """
        fig, ax = plt.subplots()
        ax.plot(range(1, len(residual_variance) + 1), residual_variance, marker='o', linestyle='-', color='b')
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('Residual Variance')
        ax.set_title('Elbow Graph of Residual Variance')
        ax.set_xticks(range(1, len(residual_variance) + 1))
        ax.grid(True)
        return fig

    
    @staticmethod
    def percentage(column_name, dataset):
        """
        Calculates the percentage for a specific column in the dataset.

        Parameters:
        -----------
        column_name : str

        Returns:
        --------
        float
            The calculated percentage.
        """
        percent = int(dataset[column_name].sum()) / 101
        return percent
    
    

    @staticmethod
    def bar_plot(perc, column_name):
        """
        Creates a bar plot showing the proportion of binary categories.

        Parameters:
        -----------
        perc : float
            The proportion for the "Yes" category.
        column_name : str
            The name of the column being visualized.

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the bar plot.
        """
        fig, ax = plt.subplots()
        ax.bar(["No", "Yes"], [1 - perc, perc], color="lightblue")
        ax.set_title(f"{column_name.capitalize()}, {round(perc, 2)}% of the dataset's animals")
        return fig