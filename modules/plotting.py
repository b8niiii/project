import streamlit as st

class Plotting:
    def __init__(self):
        pass

    def plot_organizer(list, rows, cols):
        for i in range(rows):
            cols_container = st.columns(cols)
            for j, col in enumerate(cols_container):
                index = i * cols + j
                if index < len(list):
                    with col:
                        st.pyplot(list[index])

