#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


geneDF = pd.read_csv('data.csv')
geneDF = geneDF.rename(columns = {'Unnamed: 0': 'samples'})



# look for columns with all 0's

def isZero(dataframe):
    columnsWithZeros = []
    for i in range(1, 20532):
        if not dataframe[dataframe.columns[i]].any():
            #print(dataframe.columns[i])
            columnsWithZeros.append(dataframe.columns.values[i])
    return columnsWithZeros



columnsWithZeros = isZero(geneDF)


# drop columns with zeros
geneDropped = geneDF.drop(columns = columnsWithZeros)
geneDropped.head()


# check genes
geneDF['gene_5'].unique()



# generate matrix for SVD
geneMatrix = geneDropped.drop(columns = 'samples').to_numpy()

scaler = StandardScaler()
geneMatrixScaled = scaler.fit_transform(geneMatrix)

pca = PCA(n_components=801)
Y = pca.fit(geneMatrixScaled)
var_exp = Y.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)


'''
want to find components that are causing the most variance but also find the number of dimensions 
that keep 0.8 of variance
use expplained variance ratio and put together in a plot
'''
x = ["PC%s" %i for i in range(1,40)]
trace1 = go.Bar(
    x=x,
    y=list(var_exp),
    name="Explained Variance")

trace2 = go.Scatter(
    x=x,
    y=cum_var_exp,
    name="Cumulative Variance")

layout = go.Layout(
    title='Variance',
    xaxis=dict(title='Principle Components', tickmode='linear'))

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
fig.show()



labels = pd.read_csv('labels.csv')
labels = labels.rename(columns = {'Unnamed: 0': 'sample', 'Class': 'cancer'})
labels = labels.drop(columns = ['sample'])
labels.head()



y_train = labels


# Project first three components

Y_train_pca = pca.fit_transform(geneMatrixScaled)

traces = []
for name in ['BRCA', 'KIRC', 'COAD', 'LAUD', 'PRAD']:
    trace = go.Scatter3d(
        x=Y_train_pca[y_train.cancer==name,0],
        y=Y_train_pca[y_train.cancer==name,1],
        z=Y_train_pca[y_train.cancer==name,2],
        mode='markers',
        name=name,
        marker=go.Marker(size=5, line=go.Line(width=1),opacity=1))
    
    traces.append(trace)

layout = go.Layout(
    xaxis=dict(title='PC1'),
    yaxis=dict(title='PC2'),
    title="Projection of First Three Principle Components"
)

data = traces
fig = go.Figure(data=data, layout=layout)

fig.show()



saveFile = pd.read_csv('save.txt', names = ['scoreColumn'])

saveFile.head()


saveFile['PC'] = saveFile['scoreColumn'].str.split(':').str[0]
saveFile['score'] = saveFile['scoreColumn'].str.split(':').str[-1]
saveFile.head()

saveFile2 = saveFile.drop(columns = ['scoreColumn'])
saveFile2.head(10)


saveSorted = saveFile2.sort_values(by = ['score'], ascending = False)
saveSorted['score'] = saveSorted['score'].astype('float64')


saveSorted.head(10)


saveSorted.sum()


Y_train_pca = pca.fit_transform(geneMatrixScaled)

traces = []
for name in ['BRCA', 'KIRC', 'COAD', 'LAUD', 'PRAD']: 
    trace = go.Scatter3d(
        x=Y_train_pca[y_train.cancer==name,3],
        y=Y_train_pca[y_train.cancer==name,1],
        z=Y_train_pca[y_train.cancer==name,4],
        mode='markers',
        name=name,
        marker=go.Marker(size=2, line=go.Line(width=1),opacity=1))
    
    traces.append(trace)

layout = go.Layout(
    xaxis=dict(title='PC1'),
    yaxis=dict(title='PC2'),
    title="Projection of First Three Principle Components"
)

data = traces
fig = go.Figure(data=data, layout=layout)

fig.show()


distortions = []
for i in range(1, 20):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=40, max_iter=300,
        tol=1e-04, random_state=0)
    km.fit(geneMatrix)
    distortions.append(km.inertia_)


# plot k clusters (x) against sum of squared errors within clusters (y axis)
# ideally minimum k clusters because SSE could be 0 if we put each point in its own cluster

plt.plot(range(1, 20), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors')
plt.show()


km = KMeans(
    n_clusters = 5, init = 'random',
    n_init=40, max_iter=300, 
    tol=1e-04, random_state=0)

sampleClusters = km.fit(geneMatrix)



sampleClusterLabels = sampleClusters.labels_
sampleClusterDF = pd.DataFrame(sampleClusterLabels, columns = ['labels'])
sampleClusterDF.head()


compareClusters = pd.concat([labels, sampleClusterDF], axis=1, sort=False)
#result = pd.concat([df1, df4], axis=1, sort=False)


compareClusters.head(10)





