import streamlit as st
from main import load_and_prepare_data, perform_clustering, train_classifier, compare_methods
from modules.visualization import Visualization, Plotting
from modules.classification import Animal
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd


def main():
    st.title('Animal Guesser')
    st.header("Brief Analysis on Animal Species.", divider="grey")
    st.markdown("###### The following are some interesting information extracted from a Kaggle dataset.")
    st.markdown("[Dataset link](https://www.kaggle.com/datasets/uciml/zoo-animal-classification/data)")

    st.write("")
    st.write("")
    st.write("")

    # Load and prepare data
    zoo, animal_class = load_and_prepare_data()

    # Display Box Plot
    boxplot = Visualization.plot_boxplot(zoo, 'legs', 'Legs Distribution (Box Plot)')
    st.pyplot(boxplot)

    # Display Summary Statistics
    summary = zoo['legs'].describe().round(2).to_frame().T
    st.dataframe(summary)
    st.markdown("In the box plot, we can visualize what was printed with 'describe'. It is interesting to see that the third quartile is at 4 legs, meaning that the 75% of the animals is distributed between 0 and 4 legs. The mean is 2.84")

    st.write("")
    st.write("")
    st.write("")

    legs_count = zoo['legs'].value_counts().to_dict()

    pie_chart = Visualization.plot_pie_chart(legs_count.values(), labels= legs_count.keys(), title = "Distribution of Animals by Number of Legs (Pie Chart)")
    st.pyplot(pie_chart)

    st.write("")
    st.write("")
    st.write("")
    
    columns = zoo.columns.drop(['animal_name', 'legs', 'class_type'])
    plots = []
    for element in columns:
        perc = Visualization.percentage(element, zoo)
        plots.append(Visualization.bar_plot(perc, element))
    
    choice = st.selectbox("Scegli un\' opzione: ", columns)
    indice = columns.get_loc(choice)

    Plotting.show_plot(plots, indice, figsize=(8,6))
        
    st.write("")
    st.write("")
    st.write("")

    # Perform Clustering 
    pca_data, kmeans_labels, agglomerative_labels, explained_variance = perform_clustering(
        zoo.drop(columns=['animal_name', 'class_type']))
    
    st.subheader("Developing of Unsupervised Machine Learning Methods")
    st.subheader("Agglomerative Clustering", divider = "grey")
    st.markdown("""__Agglomerative clustering__ is the type of __hierarchical clustering__ that creates clusters
                starting from the bottom __(singlular observation)__ and going upward __(merging data into clusters)__. 
                At the start, each data point has its own cluster, the process ends when we reach the requested 
                number of cluster, 7 in our case.""")
    st.markdown("""
    The agglomerative clustering can be performed with different linkage criteria:

    - **Ward**: Minimizes the variance of the clusters being merged.
    - **Average**: Uses the average of the distances of each observation of the two sets.
    - **Complete**: Uses the maximum distances between all observations of the two sets.
    - **Single**: Uses the minimum of the distances between all observations of the two sets.
    """)

    df, label_df = compare_methods(agglomerative_labels, zoo["class_type"])
    st.dataframe(df)
    predicted_columns = ["ward", "average", "single", "complete"]
    plot_list = []
    # Append confusion matrices into a list
    for col in predicted_columns:
        cm = confusion_matrix(label_df[col], label_df["actual_label"]) #ndarray
        cm_sorted = cm[np.ix_(label_df[col].value_counts().sort_values().index,label_df['actual_label'].value_counts().sort_values().index)] 
        #np.ix_ creates an indexer (mash) usefulll to perform operations
        plot = Visualization.plot_confusion_matrix(cm_sorted, "Confusion Matrix of {col}")
        plot_list.append(plot)


    Plotting.plot_organizer(plot_list, 2, 2)

    dend_plots = []
    st.subheader("Visualization with dendograms", divider = "grey")
    for col in predicted_columns:
        plot = Visualization.plot_dendrogram(col, zoo[columns], f"{col.capitalize()} Dendogram", 6)
        dend_plots.append(plot)

    Plotting.plot_organizer(dend_plots, 2, 2)

    st.header("Pca & Kmeans")
    st.subheader("KMeans Clustering Applied to Principal Components")
    residual_variance = 1 - explained_variance
    fig = Visualization.plot_elbow_graph(residual_variance)
    st.pyplot(fig)
    st.write(f"We go for 8 principal components, explaining the {round(explained_variance[7], 2)*100}% of the variance.")
    
    pca = PCA(7)
    pc = pca.fit_transform(zoo.drop(columns=['animal_name', 'class_type']))
    st.write("We use KMeans clustering method over the eight principal components to try to better classify animal species. The following confusion matrix reports the results: ")

    k_means = KMeans(n_clusters = 7, random_state = 42).fit(pc)

    cm = confusion_matrix(k_means.labels_, label_df["actual_label"])
    cm_sorted = cm[np.ix_(pd.Series(k_means.labels_).value_counts().sort_values().index ,label_df['actual_label'].value_counts().sort_values().index)]
    plot = Visualization.plot_confusion_matrix(cm_sorted, "Confusion Matrix of PCA + KMeans")
    st.pyplot(plot)

    st.markdown("We can see that this approach works better than the previous ones.")


    classifier = train_classifier(zoo)

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
    st.write("### Chosen attributes:")
    for attr in attributes:
        if st.session_state.get(attr, False):
            st.write(f"{attr}: {st.session_state[attr]}")

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
        if attr not in st.session_state: # similar to a python dictionary
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
