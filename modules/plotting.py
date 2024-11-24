import streamlit as st
import io 


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

