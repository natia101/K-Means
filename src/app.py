
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pickle

# Load the dataset

url= 'https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv'
df_raw = pd.read_csv(url)

X_raw=df_raw[['Latitude','Longitude','MedInc']]

# Scalation
sc = StandardScaler()
X = sc.fit_transform(X_raw)

# Clusterization
clf=KMeans(n_clusters=6,random_state=408) # Define model
cluster=clf.fit(X)# fit model

# Save the model as a pickle
filename = '/workspace/K-Means/models/final_model.pkl'
pickle.dump(clf, open(filename,'wb'))
