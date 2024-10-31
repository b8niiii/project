import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import streamlit as st
animal_class = pd.read_csv("animals_data\class.csv") # this is the actual division
zoo = pd.read_csv("animals_data\zoo.csv") # data we should work on 

columns = zoo.columns 
""" columns are:
'animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne',
'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous',
'fins', 'legs', 'tail', 'domestic', 'catsize', 'class_type' """

plot = zoo.boxplot("legs", grid = False, vert = False, patch_artist = True )
plt.show()


def percentage(column_name):
    zoo[column_name].sum() / len(zoo) 
for element in columns:
    perc = percentage(element)
    print(f"{element}, {perc}: of the dataset's animals ")
    plot = plt.bar(["No","Yes"],[1-perc, perc], color = 'green' )

