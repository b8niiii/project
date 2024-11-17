
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
st.header("Let's guess the animal's species")
st.subheader("This is a Subheader")
st.text("""The system is going to predict the animal species based on a 
        machine learning model. The model is a decision tree classifier trained on 
        100 animal species """)
st.markdown("Let's see how __the model__ behaves:")


columns = zoo.columns 

plot = zoo.boxplot("legs", grid = False, vert = False, patch_artist = True )
plt.show()


legs_count = zoo['legs'].value_counts().to_dict() # dictionary that counts for every possible values the occurrencies

plt.pie(legs_count.values(), labels = legs_count.keys(), autopct= '%1.0f%%',
        colors = ["#7EC8E3", 
                  "#5B92E5", 
                  "#3C69E7", 
                  "#2052B2", 
                  "#1B3A93", 
                  "#12275E"]
)
plt.show()

def percentage(column_name):
    percent = int(zoo[column_name].sum()) / 101 
    return percent

percentage("feathers")

columns = columns.drop(['animal_name','legs', 'class_type'])

plots = []
for element in columns:
    perc = percentage(element)
    
    # Create a figure and save it in the list
    fig, ax = plt.subplots()
    ax.bar(["No", "Yes"], [1 - perc, perc], color='lightblue')
    ax.set_title(f"{element.capitalize()}, {round(perc, 2)}% of the dataset's animals")
    plots.append(fig)

rows = 5
cols = 3

Plotting.plot_organizer(plots, rows, cols)


mask = zoo['catsize'] == True
print(f"{len(zoo[mask])} animals are domestic in our dataset")
zoo[~mask].head()
cat_dom = zoo[mask][zoo.domestic == True] #df of both catsize and domestic animals

per_cat_dom = len(zoo[mask][zoo.domestic == True])/len(zoo[mask]) # % of domestic animals among the cat size ones
print(f"{round(per_cat_dom, 2)}% of catsize animals are domestic in the dataset ")

"""So the 13% of the dataset refers to domestic animals and almost the 14% of the catsize animals are domestic.
Let's see among non cat size animals, we expect to have a lower percentage."""

non_cat_dom = zoo[~mask][zoo.domestic == True]
per_non_cat_dom = len(non_cat_dom)/len(zoo[~mask])
print(f"{per_non_cat_dom}% of the non catsize animals are domestic among the ones in the dataset")

"""8% of the animals are venomous, let's see among the zero legged ones:"""
mask1 = zoo['legs']== 0
zero_leg_ven = len(zoo[mask1][zoo.venomous == True])/len(zoo[mask1])
print(f"{zero_leg_ven}% of the zero legged animals are venomous.")

zoo_columns = zoo.columns
zoo_columns = zoo_columns.drop(['class_type', 'animal_name'])
df = zoo[zoo_columns] # we call df what we are going to use for the clustering algorithm

"""Agglomerative clustering is the type of hierarchical clustering that creates clusters 
starting from the bottom and going up.At the start, each data point has its own cluster, 
the process ends when we reach the requested number of clusters, 7 in our case."""

"""

The parameters of the agglomerative clustering we are interested in, are:
- n_clusters: specifies the number of clusters wanted
-   linkage: specifies the linkage criteria:
    - __ward__: minimizes the variance of the clusters being merged (__default__)
    - __average__: uses the average of the distances of each observation of the two sets
    - __complete__: uses the maximum distances between all observations of the two sets
    - __single__: uses the minimum of the distances of betewwn all observation of the two sets

"""

clustering_ward = AgglomerativeClustering(n_clusters= 7)
clustering_avg = AgglomerativeClustering(n_clusters= 7, linkage='average')
clustering_cmpl = AgglomerativeClustering(n_clusters= 7, linkage='complete')
clustering_sngl = AgglomerativeClustering(n_clusters= 7, linkage='single')

# Models Training

clustering_ward.fit(df)
clustering_avg.fit(df)
clustering_cmpl.fit(df)
clustering_sngl.fit(df)

# zoo['class_type'].value_counts().sort_values()
animal_class["Number_Of_Animal_Species_In_Class"].sort_values()

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


#Let's print the count of the values to see if we did good
print(f"Actual labels: {df['class_type'].value_counts().sort_values()}")
print(f"Ward labels: {df['ward'].value_counts().sort_values()}")
print(f"Actual labels: {df['average'].value_counts().sort_values()}")
print(f"Actual labels: {df['single'].value_counts().sort_values()}")
print(f"Actual labels: {df['complete'].value_counts().sort_values()}")


st.markdown("""####The approach above doesn't really tell much on how accurate we have been.
 ####We can use the _confusion matrix_ to confront predicted and actual values:""")

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
"""Then we can use a dendogram to visualize the division in clusters:"""
for col in predicted_columns:
    link = hierarchy.linkage(df, method= col) # Use linkage method to construct the matrix of dendogram connections
    fig, ax = plt.subplots()
    hierarchy.dendrogram(link) 
    ax.set_title(f'{col.capitalize()} Dendrogram')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Distance')
    ax.set_xticks([]) # to remove the observation's labels
    dend_plots.append(fig)


Plotting.plot_organizer(dend_plots, 2, 2)
# PCA + clustering
#Since none of the hierarchical clusterings provided the results we expected, let's try with applying the principal component analysis and k means clustering on the principal components.
#The goal of the PCA is to transform the set of correlated variables in a non-correlated variables set.
#To apply PCA we need to:
#  1. Compute the covariance matrix 
#  2. Compute egenvalues and eigenvectors to understand the direction of the principal components and
#     their strength (eigenvalues)
#  3. Create the components
#These three steps are automatically performed by scikit learn when applying pca.

# compute the covariance matrix:
pca = PCA(12)
pc = pca.fit_transform(df)
print(pc.shape)
explained_variance_cumulative = np.cumsum(pca.explained_variance_ratio_)
print(explained_variance_cumulative)

# Compute residual variance (1 - cumulative explained variance)
residual_variance = 1 - explained_variance_cumulative

# Plot of residual variance 
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(residual_variance) + 1), residual_variance, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Residual Variance')
plt.title('Elbow Graph of Residual Variance')
plt.xticks(range(1, len(residual_variance) + 1))
plt.grid(True)
plt.show()

k_means = KMeans(n_clusters= 7, random_state= 42).fit(pc)


cm = confusion_matrix(zoo['class_type'], k_means.labels_)
cm_sorted = cm[np.ix_(zoo['class_type'].value_counts().sort_values().index, pd.Series(k_means.labels_).value_counts().sort_values().index )]
fig, ax = plt.subplots()
sns.heatmap(cm_sorted, annot=True, fmt='d', cmap='viridis')  #  annot = True means that we are annotating the results inside the cells,
# fmt = d indicates the format that is decimals, cmap = viridis indicates the color of the cm
ax.set_title(f'Confusion Matrix for PCA + Cluster')
ax.set_xlabel('Predicted Cluster')
ax.set_ylabel('Actual Category')
st.pyplot(fig)


# Let's train a supervised model now to predict a new animal's class.
st.markdown("##Let's use a _supervised model_ to predict animal species: \n Here you can play with a _TREE CLASSIFIER_")
pred_df = zoo
zoo_dict = {}
for i in animal_class['Class_Number']:
    zoo_dict[i] = animal_class['Class_Type'][i-1]

 # we want to use a dictionary in order to map the elements and understand the actual lable meaning. 



pred_df['class_type'] = pred_df['class_type'].map(zoo_dict)
print(pred_df)

X = pred_df.drop(columns = ['animal_name','class_type'])
Y = pred_df['class_type']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2 ,random_state = 42)
print(X.info())

animal = Animal()

print(x_train, y_train)


classifier = AnimalClassifier()

classifier.train(X= x_train, y= y_train)

attributes = [
    "hair", "feathers", "eggs", "milk", "predator", "airborne", "toothed",
    "backbone", "breathes", "venomous", "fins", "legs", "tail", "aquatic",
    "domestic", "catsize"
]

for attr in attributes:
    if attr not in st.session_state:
        st.session_state[attr] = False


hair, feathers, eggs, milk, predator, airborne, toothed, backbone = st.columns(8)
if hair.button("Hair", use_container_width=True):
    st.session_state["hair"] = not st.session_state["hair"]
    
if feathers.button("feathers", use_container_width=True):
    st.session_state["feathers"] = not st.session_state["feathers"]
    
if eggs.button("eggs", use_container_width=True):
    st.session_state["eggs"] = not st.session_state["eggs"]
    
if milk.button("milk", use_container_width=True):
    st.session_state["milk"] = not st.session_state["milk"]
    
if predator.button("predator", use_container_width=True):
    st.session_state["predator"] = not st.session_state["predator"]
    
if airborne.button("airborne", use_container_width=True):
    st.session_state["airborne"] = not st.session_state["airborne"]
    
if toothed.button("toothed", use_container_width=True):
    st.session_state["toothed"] = not st.session_state["toothed"]
    
if backbone.button("backbone", use_container_width=True):
    st.session_state["backbone"] = not st.session_state["backbone"]
    

# Iterate over the session state and print each key-value pair

breathes, venomous, fins, legs, tail, acquatic, domestic, catsize = st.columns(8)
if breathes.button("breathes", use_container_width=True):
    st.session_state["breathes"] = not st.session_state["breathes"]
    
if venomous.button("venomous", use_container_width=True):
    st.session_state["venomous"] = not st.session_state["venomous"]
    
if fins.button("fins", use_container_width=True):
    st.session_state["fins"] = not st.session_state["fins"]
    
if legs.button("legs", use_container_width=True):
    st.session_state["legs"] = not st.session_state["legs"]
    
if tail.button("tail", use_container_width=True):
    st.session_state["tail"] = not st.session_state["tail"]
    
if acquatic.button("acquatic", use_container_width=True):
    st.session_state["aquatic"] = not st.session_state["aquatic"]
    
if domestic.button("domestic", use_container_width=True):
    st.session_state["domestic"] = not st.session_state["domestic"]
    
if catsize.button("catsize", use_container_width=True):
    st.session_state["catsize"] = not st.session_state["catsize"]
    


st.write("### Chosen attributes:")
for key, value in st.session_state.items():
    if value:
        st.write(f"{key}: {value}")

for attr in attributes:
    setattr(animal, attr, st.session_state[attr])

    
if st.button("Predict"):
    st.markdown(classifier.predict(animal=animal))
