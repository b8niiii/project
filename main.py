
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import streamlit as st
from modules.classification import Animal, AnimalClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.cluster import hierarchy 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from modules.plotting import Plotting



animal_class = pd.read_csv("animals_data\class.csv") # this is the actual division
zoo = pd.read_csv("animals_data\zoo.csv") # data we should work on 





st.title('Animal guesser')
st.header("Brief analysis on animal species.", divider = "grey")
st.markdown("###### The following are some interesting information extracted from a kaggle dataset.")
st.markdown("""[Dataset link](https://www.kaggle.com/datasets/uciml/zoo-animal-classification/data)""")
st.markdown("###### Let's start our explorative analysis by plotting some data:")


st.write("")
st.write("")
st.write("")


plot = zoo.boxplot("legs", grid = False, vert = False, patch_artist = True )
plot.set_title("Legs Distribution (Box Plot)")
fig = plot.figure # we get the figure out of the Axis object
st.pyplot(fig)

summary = round(zoo["legs"].describe(), 2).to_frame().T # we converted the series object into a dataframe and then transposed it
st.dataframe(summary)
st.markdown(""" In the boxpot we can visualize what was printed with the summary function. It is interesting to see that the third quartile is at 4 legs, meaning that the 75% of the animals is distributed between 0 and 4 legs. The mean is 2.84. """)
st.write("")
st.write("")
st.write("")

legs_count = zoo['legs'].value_counts().to_dict() # dictionary that counts the occurrencies for every possible values

fig, ax = plt.subplots()
ax.pie(legs_count.values(), labels = legs_count.keys(), autopct= '%1.0f%%',
        colors = ["#7EC8E3", 
                  "#5B92E5", 
                  "#3C69E7", 
                  "#2052B2", 
                  "#1B3A93", 
                  "#12275E"]
)
ax.set_title("Distribution of Animals by Number of Legs (Pie Chart)")
st.pyplot(fig)

def percentage(column_name):
    percent = int(zoo[column_name].sum()) / 101 
    return percent

columns = zoo.columns # Pandas uses Index objects for better performance and consistency across its DataFrames and Series, it is not a list
columns = columns.drop(['animal_name','legs', 'class_type'])

plots = []
for element in columns:
    perc = percentage(element)
    
    # Create a figure and save it in the list
    fig, ax = plt.subplots()
    ax.bar(["No", "Yes"], [1 - perc, perc], color='lightblue')
    ax.set_title(f"{element.capitalize()}, {round(perc, 2)}% of the dataset's animals")
    plots.append(fig)


# Window menu
scelta = st.selectbox('Scegli un\'opzione:', columns)
indice = columns.get_loc(scelta) # get_loc() is optimized for Index objects and avoids the overhead of conversion.

Plotting.show_plot(plots, indice, figsize=(8,6))


zoo_columns = zoo.columns
zoo_columns = zoo_columns.drop(['class_type', 'animal_name']) # we don't remove legs this time
df = zoo[zoo_columns] # we call df what we are going to use for the clustering algorithm

st.write("")
st.write("")
st.write("")

st.subheader("Developing of unsupervised machine learning methods")
st.subheader("Agglomerative Clustering", divider = "grey")
"""__Agglomerative clustering__ is the type of __hierarchical clustering__ that creates clusters 
starting from the bottom __(singular observations)__ and going up __(merging data into clusters)__.At the start, each data point has its own cluster, 
the process ends when we reach the requested number of clusters, 7 in our case."""

"""

The agglomerative clustering can be performed with different linkage criteria:
-  __ward__: minimizes the variance of the clusters being merged 
-  __average__: uses the average of the distances of each observation of the two sets
- __complete__: uses the maximum distances between all observations of the two sets
-  __single__: uses the minimum of the distances of betewwn all observation of the two sets

"""

clustering_ward = AgglomerativeClustering(n_clusters= 7, linkage = "ward")
clustering_avg = AgglomerativeClustering(n_clusters= 7, linkage='average')
clustering_cmpl = AgglomerativeClustering(n_clusters= 7, linkage='complete')
clustering_sngl = AgglomerativeClustering(n_clusters= 7, linkage='single')

# Models Training

clustering_ward.fit(df)
clustering_avg.fit(df)
clustering_cmpl.fit(df)
clustering_sngl.fit(df)

st.markdown(""" The following table shows the number of species for each class: the first column is the count of occurrencies in the actual "species" column, the latter four report the count of the four predictions.""")

#append lables to the df
ward = clustering_ward.labels_ 
average = clustering_avg.labels_ 
single = clustering_sngl.labels_ 
complete = clustering_cmpl.labels_ 
df['class_type'] = zoo['class_type'] - 1
df['ward'] = ward
df['average'] = average
df['single'] = single
df['complete'] = complete

count = pd.DataFrame()
# we use .values because value_counts() creates a Series object with values and indices
# that are automatically aligned by index, in this way we align it by value.
count["actual"] = df['class_type'].value_counts().sort_values(ascending= False).values 
count["ward"] = df['ward'].value_counts().sort_values(ascending= False).values
count["average"] = df['average'].value_counts().sort_values(ascending= False).values
count["single"] = df['single'].value_counts().sort_values(ascending= False).values
count["complete"] = df['complete'].value_counts().sort_values(ascending= False).values
count = count.reset_index(drop = True)
st.write(count)
st.markdown("""The approach above doesn't really tell much on how accurate we have been.
  We can use a __confusion matrix__ to confront predicted and actual values:""")

"""
We sort the confusion matrix by taking the index of the sorted list of value 
counts so that a __greater index__ refers to the actual __class with more observations__.
This method doesn't ensure that we are talking about the same class but is an 
effort towards that goal."""



predicted_columns = ['ward', 'average', 'single', 'complete']
# Assuming 'actual' is the column with the actual values and 'cluster_method1' is one of the clustering columns
plot_list = []
for col in predicted_columns:
    cm = confusion_matrix(df['class_type'], df[col])
    cm_sorted = cm[np.ix_(df['class_type'].value_counts().sort_values().index, df[col].value_counts().sort_values().index )]
    fig, ax = plt.subplots()
    sns.heatmap(cm_sorted, annot=True, fmt='d', cmap='viridis')  #  annot = True means that we are annotating the results inside the cells,
    # fmt = d indicates the format that is decimals, cmap = viridis indicates the color of the cm
    ax.set_title(f'Confusion Matrix for {col}')
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('Actual Category')
    plot_list.append(fig)

Plotting.plot_organizer(plot_list, 2, 2)

dend_plots =[]
st.subheader("Dendogram visualization", divider = "grey")
"""Then we can use dendograms to visualize the division in clusters:"""
for col in predicted_columns:
    link = hierarchy.linkage(df, method= col) # Use linkage method to construct the matrix of dendogram connections
    # Determine the threshold to stop at 7 clusters
    # This finds the distance where the number of clusters is exactly 7
    cluster_threshold = sorted(link[:, 2], reverse=True)[6]  # 6th largest linkage distance corresponds to 7 clusters

    fig, ax = plt.subplots()
    hierarchy.dendrogram(link) 
    ax.axhline(y=cluster_threshold, color='r', linestyle='--', label=f'7 Clusters Cut-Off')
    ax.set_title(f'{col.capitalize()} Dendrogram')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Distance')
    ax.set_xticks([]) # to remove the observation's labels
    dend_plots.append(fig)


Plotting.plot_organizer(dend_plots, 2, 2)

# compute the covariance matrix:
st.write("")
st.write("")
st.write("")
st.subheader("Principal Component Analysis and K-Means", divider = "grey")
pca = PCA(12)
pc = pca.fit_transform(df)

explained_variance_cumulative = np.cumsum(pca.explained_variance_ratio_)


# Compute residual variance (1 - cumulative explained variance)
residual_variance = 1 - explained_variance_cumulative
st.write("Let's use the eblow graph of the residual variance to choose the perfect number of components for our use case.")
# Plot of residual variance 
fig, ax = plt.subplots()
#plt.figure(figsize=(8, 5))
ax.plot(range(1, len(residual_variance) + 1), residual_variance, marker='o', linestyle='-', color='b')
#plt.plot(range(1, len(residual_variance) + 1), residual_variance, marker='o', linestyle='-', color='b')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Residual Variance')
ax.set_title('Elbow Graph of Residual Variance')
ax.set_xticks(range(1, len(residual_variance) + 1))
ax.grid(True)
st.pyplot(fig)

st.write(f"We go for 8 principal components, explaining the {round(explained_variance_cumulative[7], 2)*100}% of the variance.")

pca = PCA(8)
pc = pca.fit_transform(df)

st.write("We use a KMeans clustering method over the seven principal components with  to try to better classify animal species. The following is the confusion matrix with the results:")
k_means = KMeans(n_clusters= 7, random_state= 42).fit(pc)


cm = confusion_matrix(zoo['class_type'], k_means.labels_)
cm_sorted = cm[np.ix_(zoo['class_type'].value_counts().sort_values().index, pd.Series(k_means.labels_).value_counts().sort_values().index )]
fig, ax = plt.subplots()
sns.heatmap(cm_sorted, annot=True, fmt='d', cmap='viridis')  #  annot = True means that we are annotating the results inside the cells,
# fmt = d indicates the format that is decimals, cmap = viridis indicates the color of the cm
ax.set_title(f'Confusion Matrix for PCA & KMeans')
ax.set_xlabel('Predicted Cluster')
ax.set_ylabel('Actual Category')
st.pyplot(fig)


# Let's train a supervised model now to predict a new animal's class.
st.subheader("Let's use a __supervised model__ to predict animal species:", divider = "grey")
st.write("Here you can play with a __TREE CLASSIFIER__ that has been trained on the original dataset. \n Choose your animal's attributes and the model wills spit out its species. ")
pred_df = zoo
zoo_dict = {}
for i in animal_class['Class_Number']:
    zoo_dict[i] = animal_class['Class_Type'][i-1]

 # we want to use a dictionary in order to map the elements and understand the actual lable meaning. 



pred_df['class_type'] = pred_df['class_type'].map(zoo_dict)
print(pred_df)

X = pred_df.drop(columns = ['animal_name','class_type'])
Y = pred_df['class_type']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.01 ,random_state = 42)
print(X.info())

animal = Animal()

print(x_train, y_train)


classifier = AnimalClassifier()

classifier.train(X= x_train, y= y_train)


attributes = [
    "hair", "feathers", "eggs", "milk", "predator", "airborne", "toothed",
    "backbone", "breathes", "venomous", "fins", "tail", "aquatic",
    "domestic", "catsize", "legs"
]

# Initialize session state for each attribute
for attr in attributes:
    if attr not in st.session_state:
        st.session_state[attr] = False

# Create buttons for attributes
hair, feathers, eggs, milk = st.columns(4)
if hair.button("Hair", use_container_width=True):
    st.session_state["hair"] = not st.session_state["hair"]
    
if feathers.button("feathers", use_container_width=True):
    st.session_state["feathers"] = not st.session_state["feathers"]
    
if eggs.button("eggs", use_container_width=True):
    st.session_state["eggs"] = not st.session_state["eggs"]
    
if milk.button("milk", use_container_width=True):
    st.session_state["milk"] = not st.session_state["milk"]

predator, airborne, toothed, backbone  = st.columns(4)
if predator.button("predator", use_container_width=True):
    st.session_state["predator"] = not st.session_state["predator"]
    
if airborne.button("airborne", use_container_width=True):
    st.session_state["airborne"] = not st.session_state["airborne"]
    
if toothed.button("toothed", use_container_width=True):
    st.session_state["toothed"] = not st.session_state["toothed"]
    
if backbone.button("backbone", use_container_width=True):
    st.session_state["backbone"] = not st.session_state["backbone"]
    
breathes, venomous, fins, tail = st.columns(4)
if breathes.button("breathes", use_container_width=True):
    st.session_state["breathes"] = not st.session_state["breathes"]
    
if venomous.button("venomous", use_container_width=True):
    st.session_state["venomous"] = not st.session_state["venomous"]
    
if fins.button("fins", use_container_width=True):
    st.session_state["fins"] = not st.session_state["fins"]
  
if tail.button("tail", use_container_width=True):
    st.session_state["tail"] = not st.session_state["tail"]

acquatic, domestic, catsize = st.columns(3)    
if acquatic.button("acquatic", use_container_width=True):
    st.session_state["aquatic"] = not st.session_state["aquatic"]
    
if domestic.button("domestic", use_container_width=True):
    st.session_state["domestic"] = not st.session_state["domestic"]
    
if catsize.button("catsize", use_container_width=True):
    st.session_state["catsize"] = not st.session_state["catsize"]

# Slider for legs
left, middle, centre = st.columns(3) 
selected_legs = middle.slider("legs", 0, 8, 1)
st.session_state["legs"] = selected_legs


for attr in attributes:
    setattr(animal, attr, st.session_state[attr])

# Display chosen attributes in original order
st.subheader("Chosen attributes:", divider = "grey")
for attr in attributes:
    if st.session_state.get(attr, False):
        st.write(f"{attr}: {st.session_state[attr]}")

# Predict button
if st.button("Predict"):
    st.markdown(classifier.predict(animal=animal))
