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
        ax.pie(values, labels=labels, autopct='%1.0f%%', colors = ['#7EC8E3', '#5B92E5', '#3C69E7', '#2052B2','#1B3A93','#12275E'])
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
    def plot_dendrogram(method,df, title, cluster_threshold=None):
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
       percent = int(dataset[column_name].sum())/101
       return percent
    
    
    @staticmethod
    def bar_plot(perc, column_name):
        fig, ax = plt.subplots()
        ax.bar(["No", "Yes"], [1-perc, perc], color = "lightblue")
        ax.set_title(f"{column_name.capitalize()}, {round(perc, 2)}% of the dataset's animals")
        return fig