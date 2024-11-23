# app.py

import streamlit as st
from main import load_and_prepare_data, perform_clustering, train_classifier
from modules.visualization import Visualization
from modules.classification import Animal
from modules.plotting import Plotting

def main():
    st.title('Animal Guesser')
    st.header("Brief Analysis on Animal Species.", divider="grey")
    st.markdown("###### The following are some interesting information extracted from a Kaggle dataset.")
    st.markdown("[Dataset link](https://www.kaggle.com/datasets/uciml/zoo-animal-classification/data)")

    # Load and prepare data
    zoo, animal_class = load_and_prepare_data()

    # Display Box Plot
    boxplot_fig = Visualization.plot_boxplot(zoo, 'legs', 'Legs Distribution (Box Plot)')
    st.pyplot(boxplot_fig)

    # Display Summary Statistics
    summary = zoo['legs'].describe().round(2).to_frame().T
    st.dataframe(summary)
    st.markdown("In the box plot, we can visualize the summary statistics for the number of legs.")

    # Perform Clustering (if needed for visualization)
    pca_data, kmeans_labels, agglomerative_labels = perform_clustering(
        zoo.drop(columns=['animal_name', 'class_type']))

    # Visualize Clustering Results
    # ... (Use your Visualization and Plotting classes)

    # Train Classifier
    classifier = train_classifier(zoo, animal_class)

    # Animal Attribute Selection
    st.subheader("Predict Animal Class", divider="grey")
    st.write("Choose your animal's attributes, and the model will predict its class.")

    attributes = [
        "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
        "toothed", "backbone", "breathes", "venomous", "fins",
        "tail", "domestic", "catsize", "legs"
    ]

    animal = Animal()
    create_attribute_inputs(animal, attributes)

    # Predict Button
    if st.button("Predict"):
        prediction = classifier.predict(animal)
        st.markdown(f"### Predicted Class: {prediction}")

def create_attribute_inputs(animal, attributes):
    """
    Creates input widgets for each animal attribute.
    """
    # Initialize session state for each attribute
    for attr in attributes:
        if attr not in st.session_state:
            st.session_state[attr] = False

    # Create input widgets
    cols = st.columns(4)
    for idx, attr in enumerate(attributes):
        if attr != 'legs':
            with cols[idx % 4]:
                st.session_state[attr] = st.checkbox(attr.capitalize(), value=st.session_state[attr])
                setattr(animal, attr, st.session_state[attr])
        else:
            st.session_state['legs'] = st.slider("Legs", 0, 8, value=2)
            animal.legs = st.session_state['legs']

if __name__ == '__main__':
    main()
