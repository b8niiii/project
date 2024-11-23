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
        fig, ax = plt.subplots()
        data.boxplot(column=column, grid=False, vert=False, patch_artist=True, ax=ax)
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_pie_chart(values, labels, title):
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.0f%%')
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_confusion_matrix(cm, title):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        return fig

    @staticmethod
    def plot_dendrogram(linkage_matrix, title, cluster_threshold=None):
        fig, ax = plt.subplots()
        hierarchy.dendrogram(linkage_matrix, ax=ax)
        if cluster_threshold:
            ax.axhline(y=cluster_threshold, color='r', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Distance')
        return fig

    @staticmethod
    def plot_elbow_graph(residual_variance):
        fig, ax = plt.subplots()
        ax.plot(range(1, len(residual_variance) + 1), residual_variance, marker='o', linestyle='-', color='b')
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('Residual Variance')
        ax.set_title('Elbow Graph of Residual Variance')
        ax.set_xticks(range(1, len(residual_variance) + 1))
        ax.grid(True)
        return fig
