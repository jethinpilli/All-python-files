# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:10:46 2020

@author: DELL
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


# If .csv file is given read that as X 
# do X=X.as_matrix
#and do steps make_blobs but generally it is not given since clustering is for unlabelled data

np.random.seed(0)
X,y=make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2,3], [1,1]], cluster_std=0.9)

plt.scatter(X[:,0], X[:,1], marker='.')

k_means=KMeans(init="k-means++", n_clusters=4, n_init=12)

k_means.fit(X)

k_means_labels=k_means.labels_
k_means_labels

k_means_cluster_centers=k_means.cluster_centers_
k_means_cluster_centers

#Initialize the plot with the specified dimensions
fig=plt.figure(figsize=(6,4))

#Colours us a colour map, which will produce an array of coloures based on
#the number of labels that are. We use set(k_means_labels) to ge the unique labels
colors=plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))

#create a plot
ax=fig.add_subplot(1,1,1)

# For loops that plots the data points and centroids
#k will range from 0-3, which will match the possible clusters that each
# Labeled as false

for k, col in zip(range(len([[4,4], [-2,-1], [2,-3], [1,1]])), colors):
    
    #create a list of all data points, where the data points that are
    #in the cluster (ex.cluster 0) are labeled as true, else they are
    #labeled as false.
    
    my_members= (k_means_labels ==k)
    
    # Define the centriod, or the center cluster
    cluster_center=k_means_cluster_centers[k]
    
    #Plot the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members,1], 'w', markerfacecolor=col, marker='.')
    
    #Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    
#Title of the plot
ax.set_title("KMeans")

#Remove x-axis and y-acis ticks
ax.set_xticks(())
ax.set_yticks(())

plt.show()    