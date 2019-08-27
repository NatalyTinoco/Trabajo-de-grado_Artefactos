# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:41:24 2019

@author: Nataly
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

datos=pd.read_excel('moviescs.xlsx')
df=pd.DataFrame(datos)
x=df['cast_total_facebook_likes']
y=['imdb_score']

X=df[['cast_total_facebook_likes','imdb_score']].as_matrix()

#X=np.array(list(zip(x,y)))


kmeans=KMeans(n_clusters=2)
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

colors=["m.","r.",".c","y.","b."]
for i in range(len(X)):
    print('Coordenada: ',X[i],'Etiqueta: ',labels[i])
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)

plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=150,linewidths=5,zorder=10)
plt.show()


#estimator





