import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Unblock?
#from kneed import KneeLocator
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import ClusterHelper as ch
import importlib
import plotly.express as px
import plotly.graph_objects as go
importlib.reload(ch)
import streamlit as st

#Data Preparation

df = pd.read_csv(r'acs_demographic_data_by_census_tract.csv')
output_df = ch.prepareDataOneProduct(df)

#What are we looking at
'''The dataset contains census data of different areas in the United States, in our case each row of data contains information about the demography differences of stores. In addition we have sales data for each store at each location and for each product. The list below is a list of available data points for each product and store combination. 

  - Demographic data around each store, with Sales data from every store.
  - Demographics are made for example: median income, pct. employed etc.
  - In addition, Sales number for each store and for each product is used.
  - As an example, we use Rema 1000 and Kiwi stores in Oslo
'''

#Data Cleaning
'''
Data cleaning is a very extraneous process, and where most of the effort in an unsupervised learning task lays. If you were to divide the time spent on tasks in a machine learning development cycle, 90 percent would be spent on this point. Since we are here looking at a synthetic dataset, the data is quite clean and easy to use. If you were to use real data this would be much more difficult. A point which is essential here is that when doing a large parallell programming task, the outlier detection has to be programatically defined. The difficulty there lies in what is a good outlier(store outperforming it's cluster average) vs. what is not(data errors etc.). This has to be coordinated with PMs. Some effects of bad data cleaning is shown under.  
'''

#Effects of Outliers
'''
First, let's see what happens when outliers are allowed to persist in the dataset. 
'''
kmeans_set = output_df.copy()
kmeans_set = ch.outlierAddition(kmeans_set)
kmeans_set = kmeans_set.loc[(kmeans_set!=0).any(axis=1)]

#Init sklearn objects
Sc = StandardScaler()
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init='auto',
    max_iter=3000
)
#Run pca to find plottable data
pca = PCA(n_components=3)

#Fit pandas dataframe
X = Sc.fit_transform(kmeans_set)

#Fit PCA and kmeans
kmeans.fit(X)
pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2', 'PC3'])
pca_data['cluster'] = pd.Categorical(kmeans.labels_)

#Clusters plotted against PCA axises
#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], s = 40, alpha = 1, c= pca_data['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Kmeans With Outliers')
# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()
import mpld3
import streamlit.components.v1 as components
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600)
st.pyplot(fig)


#What happens when outliers are removed from dataset
'''
Initially, when we plotted the clusters here, we did not do any outlier removal. Now, by removing outliers we can see that some of the points previously contained in "wrong" clusters are now in the correct clusters. That means we loose some stores in the clustering, but the stores remaining are more correctly clustered.

A consideration that needs to be made here is what makes an outlier in this specific case, is a store outperforming other stores an outlier that should be excluded or included?
'''

kmeans_set_without_outliers = output_df.copy()



#Init sklearn objects
Sc = StandardScaler()
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init='auto',
    max_iter=3000
)
#Run pca to find plottable data
pca = PCA(n_components=3)

#Fit pandas dataframe
X = Sc.fit_transform(kmeans_set_without_outliers)


#Fit PCA and kmeans
kmeans.fit(X)
pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2', 'PC3'])
pca_data['cluster'] = pd.Categorical(kmeans.labels_)
pca_data_plot = pca_data.copy()
pca_data_plot.loc[pca_data['cluster'] == 0,'cluster'], pca_data_plot.loc[pca_data['cluster'] == 1,'cluster'], pca_data_plot.loc[pca_data['cluster'] == 2,'cluster'] = 2, 1, 0 

#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(-1*pca_data_plot['PC1'], pca_data_plot['PC2'], pca_data_plot['PC3'], s = 40, alpha = 1, c= pca_data_plot['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')


# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()

#What does a cluster consist of
'''
Similarly to the explanation in the presentation, each cluster represents a user group. In our case for example: 
- Cluster 0: Low income area, consisting mostly of students aged 20-30.
- Cluster 1: Medium income area, consisting of families with children.
- Cluster 2: Older couples aged 60 and above.
'''

#print(kmeans_set_without_outliers.head())

#DBSCAM
'''
DBSCAN is an alternative clustering method to Kmeans clustering. One of the issues with Kmeans clustering is the sensitivity to ouliers. DBSCAN tries to solve this by introducing a maximal nearest neighbour distance epsilon. Points further away than epsilon from a neighbouring point is marked as an outlier. -1 in the graph below. 
'''
dbscan = DBSCAN(eps = 5.25, min_samples= 2).fit(X)
pca_data['cluster'] = pd.Categorical(dbscan.labels_)

#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], s = 40, alpha = 1, c= pca_data['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()

'''
As seen in the figure, there are some issues with DBSCAN, namely epsilon optimization and the fact that it does not take into account that clusters may have varying density of points. As seen in the picture below
'''

#OPTICS
'''
Ordering points to identify the clustering structure. A method that tries to take into account that clusters may have varying density. It does so by introducing a local distance measure instead of the epsilon in the previous method.
'''

optics = OPTICS(min_samples=2).fit(X)
pca_data['cluster'] = pd.Categorical(dbscan.labels_)

#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], s = 40, alpha = 1, c= pca_data['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()

#Temporal dependance in Sales
'''
It is a good assumption that products sell differently in different time periods over a year. So a good clustering method needs to take into account when the sales of the store is underperforming it's group. 
'''

#Multiple products at once

#Apply time series data
df = pd.read_csv(r'acs_demographic_data_by_census_tract.csv')
groupedProductData = ch.createBulkData(df)

#What does a cluster consist of
#print(kmeans_set_without_outliers.head())

'''
- Cluster 0: Low income area, small/medium stores, consisting mostly of students aged 20-30, Politics: 30% AP, 10% V, 25 MDG, 15% H.
- Cluster 1: Medium income area, large stores, consisting of families with children, 86% employment rate, Politics: 30% H, 20% AP, 5% V.
- Cluster 2: Older couples aged 60 and above, medium/large stores, 38% employment rate, median income 450k NOK.
'''

importlib.reload(ch)
#UNDO THIS COMMENT
#ch.clusterSpecificProduct(groupedProductData, 0)

'''
- Cluster 0: Low income area, consisting mostly of students aged 20-30, Politics: 30% AP, 10% V, 25 MDG, 15% H.
- Cluster 1: Low income area, low emplyment rate 23%, Politics: 30% FRP, 20% AP, 5% V.
- Cluster 2: Affluent area, median income above 1.2M NOK.
- Cluster 3: Medium income area, consisting of families with children, 86% employment rate, Politics: 30% H, 20% AP, 5% V.
- Cluster 4: Older couples aged 60 and above, 38% employment rate, median income 450k NOK.
'''
#UNDO THIS COMMENT
#ch.clusterSpecificProduct(groupedProductData, 1)

# Considerations for parallelizasion of clusters
'''
* Outliers semi-manually evaluated per group
* Number of Clusters evalulated on a Category basis
* Different types of clustering algorithms benchmarked against each other per category
* Data cleaning necessary at scale
'''



st.title("This is a Clustering Demo with maps!")

st.plotly_chart(ch.clusterSpecificProduct(groupedProductData, 0))

st.plotly_chart(ch.clusterSpecificProduct(groupedProductData, 1))



