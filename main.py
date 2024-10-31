import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st
animal_class = pd.read_csv("animals_data\class.csv") # this is the actual division
zoo = pd.read_csv("animals_data\zoo.csv") # data we should work on 

columns = zoo.columns 
""" columns are:
'animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne',
'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous',
'fins', 'legs', 'tail', 'domestic', 'catsize', 'class_type' """

plot = zoo.boxplot("legs", grid = False, vert = True, patch_artist = True )
plt.show()




