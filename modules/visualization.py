import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
import streamlit as st
import io 

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
class Plotting:
    """
    A class for organizing and displaying plots in a Streamlit app.

    This class provides utilities to arrange and display multiple plots in a grid format 
    or individually. It also supports downloading plots directly from the app.
    """

    @staticmethod
    def plot_organizer(list, rows, cols):
        """
        Organizes and displays a grid of plots in a Streamlit app.

        This method arranges plots into a grid with the specified number of rows and columns 
        and displays them using Streamlit's layout system.

        Args:
            list (list): A list of plots (e.g., matplotlib figures) to display.
            rows (int): The number of rows in the grid layout.
            cols (int): The number of columns in the grid layout.

        Functionality:
            - Creates a grid layout using Streamlit's `st.columns`.
            - Displays plots in individual cells of the grid, if available.
            - Handles cases where the number of plots is less than the total grid cells.

        Example:
            >>> plots = [fig1, fig2, fig3, fig4]  # List of matplotlib figures
            >>> Plotting.plot_organizer(plots, rows=2, cols=2)
        """
        for i in range(rows):
            cols_container = st.columns(cols)
            for j, col in enumerate(cols_container):
                index = i * cols + j
                if index < len(list):
                    with col:
                        st.pyplot(list[index])

    @staticmethod
    def show_plot(list: list, i:int, figsize = (8,6), download_label = "Download Plot"):
        """
        Displays a single plot from a list of plots in a Streamlit app.

        This method resizes the selected plot, displays it in the app, and provides 
        a download button for users to save the plot as a PNG file.

        Args:
            list (list): A list of plots (e.g., matplotlib figures).
            i (int): The index of the plot to display from the list.
            figsize (tuple, optional): A tuple specifying the width and height of the plot 
                                       in inches. Defaults to (8, 6).
            download_label (str, optional): The label for the download button. 
                                            Defaults to "Download Plot".

        Functionality:
            - Resizes the plot to the specified dimensions using `set_size_inches`.
            - Saves the plot to an in-memory buffer as a PNG image.
            - Adds a download button for users to save the plot with the specified file name.
            - Displays the plot in the Streamlit app.

        Notes:
            - The buffer (`io.BytesIO`) acts as temporary storage for the plot image in memory.
            - `buf.seek(0)` resets the cursor in the buffer to allow the image to be read from the start.
            - The MIME type `image/png` ensures the file is recognized as a PNG image.

        Example:
            >>> plots = [fig1, fig2, fig3]  # List of matplotlib figures
            >>> Plotting.show_plot(plots, i=0, figsize=(10, 5), download_label="Save This Plot")
        """

        list[i].set_size_inches(figsize)

        buf = io.BytesIO() # buffer
        list[i].savefig(buf, format="png")
        buf.seek(0) # reset the cursor
        st.download_button(
            label=download_label,
            data=buf,
            file_name=f"plot_{i}.png",
            mime="image/png",) # is a standardized way to indicate the type of a file, image/png means png.

        st.pyplot(list[i])

