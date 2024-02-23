import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import importlib
import plotly.express as px
import plotly.graph_objects as go
np.random.seed(42)

def generate_product_sales(row):
    relevant_columns = [
    'CENSUS_TRACT',
    'TOTAL_HOMES',
    'TOTAL_OWNED',
    'PCT_OWNED_OF_TOTAL',
    '15-34_pct_of_owned',
    '35-64_pct_of_owned',
    '65+_pct_of_owned',
    '2017+_pct_of_owned',
    '2015-16_pct_of_owned',
    '2010-14_pct_of_owned',
    '2000-09_pct_of_owned',
    '1990-99_pct_of_owned',
    '1989-_pct_of_owned',
    'MEDIAN_AGE_ALL_TOTAL',
    'MEDIAN_AGE_ALL_MALES',
    'MEDIAN_AGE_ALL_FEMALES',
    'MEDIAN_AGE_NATIVE_TOTAL',
    'MEDIAN_AGE_NATIVE_MALES',
    'MEDIAN_AGE_NATIVE_FEMALES',
    'MEDIAN_AGE_FOREIGN_BORN_TOTAL',
    'MEDIAN_AGE_FOREIGN_BORN_MALES',
    'MEDIAN_AGE_FOREIGN_BORN_FEMALES',
    'MEDIAN_AGE_WORKERS_TOTAL',
    'MEDIAN_AGE_WORKERS_MALES',
    'MEDIAN_AGE_WORKERS_FEMALES',
    'TOTAL_POPULATION',
    'TOTAL_INCOME',     
    'TOTAL_INCOME_PER_CAP',
    'AVG_COMMUTE_IN_MINUTES',
    'PCT_VOTING_AGE_CITIZENS',
    'PCT_EMPLOYED',
    'PCT_MEN',
    'PCT_POVERTY_ALL',
    'PCT_POVERTY_CHILD',
    'FIELD_PCT_PROFESSIONAL',
    'FIELD_PCT_SERVICE',
    'FIELD_PCT_OFFICE',
    'FIELD_PCT_CONSTRUCTION',
    'FIELD_PCT_PRODUCTION',
    'COMMUTE_PCT_DRIVE',
    'COMMUTE_PCT_CARPOOL',
    'COMMUTE_PCT_TRANSIT',
    'COMMUTE_PCT_WALK',
    'COMMUTE_PCT_OTHER',
    'COMMUTE_PCT_WORK_FROM_HOME',
    'WORK_PCT_PRIVATE',
    'WORK_PCT_PUBLIC',
    'WORK_PCT_SELF_EMPLOYED',
    'WORK_PCT_UNEMPLOYED']
    # Randomly generate weights for each column
    weights = np.random.uniform(0, 0.1, len(relevant_columns))
    # Introduce different centers for each product category
    category_centers = [15000, 25000, 40000]  # You can customize these centers based on your needs
    center = category_centers[int(row['Product Category'])]
    
    sales = center + 0.1*np.random.normal(loc = row['TOTAL_INCOME'],scale = 0.05*row['TOTAL_INCOME']) +  np.random.normal(loc=0, scale=1000) + np.random.normal(loc=row['PCT_EMPLOYED']*1000, scale=0.05*row['PCT_EMPLOYED']*1000)
    return sales

def generate_product_sales_temporal(row):

    relevant_columns = [
    'CENSUS_TRACT',
    'TOTAL_HOMES',
    'TOTAL_OWNED',
    'PCT_OWNED_OF_TOTAL',
    '15-34_pct_of_owned',
    '35-64_pct_of_owned',
    '65+_pct_of_owned',
    '2017+_pct_of_owned',
    '2015-16_pct_of_owned',
    '2010-14_pct_of_owned',
    '2000-09_pct_of_owned',
    '1990-99_pct_of_owned',
    '1989-_pct_of_owned',
    'MEDIAN_AGE_ALL_TOTAL',
    'MEDIAN_AGE_ALL_MALES',
    'MEDIAN_AGE_ALL_FEMALES',
    'MEDIAN_AGE_NATIVE_TOTAL',
    'MEDIAN_AGE_NATIVE_MALES',
    'MEDIAN_AGE_NATIVE_FEMALES',
    'MEDIAN_AGE_FOREIGN_BORN_TOTAL',
    'MEDIAN_AGE_FOREIGN_BORN_MALES',
    'MEDIAN_AGE_FOREIGN_BORN_FEMALES',
    'MEDIAN_AGE_WORKERS_TOTAL',
    'MEDIAN_AGE_WORKERS_MALES',
    'MEDIAN_AGE_WORKERS_FEMALES',
    'TOTAL_POPULATION',
    'TOTAL_INCOME',     
    'TOTAL_INCOME_PER_CAP',
    'AVG_COMMUTE_IN_MINUTES',
    'PCT_VOTING_AGE_CITIZENS',
    'PCT_EMPLOYED',
    'PCT_MEN',
    'PCT_POVERTY_ALL',
    'PCT_POVERTY_CHILD',
    'FIELD_PCT_PROFESSIONAL',
    'FIELD_PCT_SERVICE',
    'FIELD_PCT_OFFICE',
    'FIELD_PCT_CONSTRUCTION',
    'FIELD_PCT_PRODUCTION',
    'COMMUTE_PCT_DRIVE',
    'COMMUTE_PCT_CARPOOL',
    'COMMUTE_PCT_TRANSIT',
    'COMMUTE_PCT_WALK',
    'COMMUTE_PCT_OTHER',
    'COMMUTE_PCT_WORK_FROM_HOME',
    'WORK_PCT_PRIVATE',
    'WORK_PCT_PUBLIC',
    'WORK_PCT_SELF_EMPLOYED',
    'WORK_PCT_UNEMPLOYED']
    # Introduce different centers for each product category
    category_centers = [15000, 25000, 40000]  # You can customize these centers based on your needs
    center = category_centers[int(row['Product Category'])]
    weights = np.array([0.2,0.35,0.5,0.7,0.6,0.8,1.2,0.9,0.7,0.4,0.1,0.1])
    sales = weights[int(row['MONTH'])] * (center + 0.1*np.random.normal(loc = row['TOTAL_INCOME'],scale = 0.05*row['TOTAL_INCOME']) +  np.random.normal(loc=0, scale=1000) + np.random.normal(loc=row['PCT_EMPLOYED']*1000, scale=0.05*row['PCT_EMPLOYED']*1000))
    return sales

def prepareDataOneProduct(df):
    df = df.rename(columns= {'STATE' : 'STORE'})
    numerical_states = np.linspace(0,49,50)
    # Identify numeric columns for mean calculation
    numeric_columns = df.select_dtypes(include='number').columns

    # Group by 'Category' and calculate the mean for all numeric columns
    grouped_df = df.groupby('STORE')[numeric_columns].mean().reset_index()

    #Makes states numerical
    grouped_df['STORE'] = numerical_states

    #Add product categories and total sales
    product_0_df = grouped_df.copy()
    product_0_df['Product Category'] = 0
    

    product_1_df = grouped_df.copy()
    product_1_df['Product Category'] = 1

    product_2_df = grouped_df.copy()
    product_2_df['Product Category'] = 2

    expanded_df = pd.concat([product_0_df, product_1_df, product_2_df], ignore_index=True)
    

    output_df = expanded_df.copy()
    #df.loc[df['Product Category'] == 0]
    output_df = output_df[output_df['Product Category'] ==0]

    return output_df

def prepareTemporalData(df, nProducts):
    df = df.rename(columns={'STATE' : 'STORE'})
    numerical_states = np.linspace(0,49,50)
    # Identify numeric columns for mean calculation
    numeric_columns = df.select_dtypes(include='number').columns

    # Group by 'Category' and calculate the mean for all numeric columns
    grouped_df = df.groupby('STORE')[numeric_columns].mean().reset_index()

    #Makes states numerical
    grouped_df['STORE'] = numerical_states

    #Add product categories and total sales
    product_0_df = grouped_df.copy()
    product_0_df['Product Category'] = 0
    

    product_1_df = grouped_df.copy()
    product_1_df['Product Category'] = 1

    product_2_df = grouped_df.copy()
    product_2_df['Product Category'] = 2

    expanded_df = pd.concat([product_0_df, product_1_df, product_2_df], ignore_index=True)
    
    df_list = []
    for i in range(12):
        month_frame = expanded_df.copy()
        month_frame['MONTH'] = i
        df_list.append(month_frame)
    expanded_df = pd.concat(df_list)        
    expanded_df['Product Sales'] = expanded_df.apply(generate_product_sales_temporal, axis=1)

    output_df = expanded_df.copy()
    #df.loc[df['Product Category'] == 0]
    
    output_df = output_df[output_df['Product Category'] ==0]
    return output_df

def createBulkData(df):
    df = df.rename(columns={'STATE' : 'STORE'})
    numerical_states = np.linspace(0,49,50)
    # Identify numeric columns for mean calculation
    numeric_columns = df.select_dtypes(include='number').columns

    # Group by 'Category' and calculate the mean for all numeric columns
    grouped_df = df.groupby('STORE')[numeric_columns].mean().reset_index()

    #Makes states numerical
    grouped_df['STORE'] = numerical_states

    #Add product categories and total sales
    product_0_df = grouped_df.copy()
    product_0_df['Product Category'] = 0
    

    product_1_df = grouped_df.copy()
    product_1_df['Product Category'] = 1

    product_2_df = grouped_df.copy()
    product_2_df['Product Category'] = 2
    OutputList = [product_0_df, product_1_df, product_2_df]
    for i in range(len(OutputList)):
        df_list = []
        for j in range(12):
            month_frame = OutputList[i].copy()
            month_frame['MONTH'] = j
            df_list.append(month_frame)
        OutputList[i] = pd.concat(df_list)        
        OutputList[i]['Product Sales'] = OutputList[i].apply(generate_product_sales_temporal, axis=1)

    return OutputList

def outlierAddition(df):
    #Add outliers here
    # List of the averages for the test. 
    averages = [df[key].describe()['mean'] for key in df]
    stds = [df[key].describe()['std'] for key in df]

    indexes = df.index.tolist()
    df.reindex(indexes)
    # Adding the mean row to the bottom of the 
    #print(output_df.at['51', 'Product Sales'])

    i = 0
    for key in df:
        if(key != 'Product Category' and key != 'Product Sales'):
            df.at[25, key] =  averages[i]+2*stds[i]    
            df.at[40, key] =  averages[i]-2*stds[i]
            i += 1
    #Add 0 values, difficulty that can be explained is in terms of when are there NaN values, or 0. 
    #The easy case done here is that all or none in a row is zero, if they are more sparse some sort of selection criteria is needed, which might be hard when columns differ
    n_blank = 2
    new_index = pd.RangeIndex(len(df)*(n_blank+1))
    new_df = pd.DataFrame(0, index=new_index, columns=df.columns)
    ids = np.arange(len(df))*(n_blank+1)
    new_df.loc[ids] = df.values
    df = new_df
    df = df[df['Product Category'] == 0]
    return df



def elbowMethod(X):
    distortions = []
    for k in range(2,20):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), distortions)
    plt.grid(True)
    plt.xlabel('N Clusters')
    plt.ylabel('MSE')
    plt.title('Elbow curve')

def correlationPlotter(df):
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')

def applyClusterLabels(df, clusterLabels, product):
    clusterArr = clusterLabels.to_numpy()
    clusterArr = clusterArr.flatten()
    clusterArr = np.tile(clusterArr,12)
    
    df['Cluster Id'] = clusterArr
    if product == 0:
        df.loc[df['Cluster Id'] == 0, 'Product Sales'] = df.loc[df['Cluster Id'] == 0, 'Product Sales']*0.7
        df.loc[df['Cluster Id'] == 1, 'Product Sales'] = df.loc[df['Cluster Id'] == 1, 'Product Sales']*0.6
        df.loc[(df['STORE'] == 31.0) & (df['MONTH'] == 8), 'Product Sales'] = df.loc[(df['STORE'] == 31.0) & (df['MONTH'] == 8), 'Product Sales']*2
        #Demo for Month 8-10 Grunerlokka
        df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 8), 'Product Sales'] = df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 8), 'Product Sales']*5
        df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 9), 'Product Sales'] = df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 9), 'Product Sales']*5
        df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 10), 'Product Sales'] = df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 10), 'Product Sales']*0.25
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 8), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 8), 'Product Sales']*5
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 9), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 9), 'Product Sales']*5
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 10), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 10), 'Product Sales']*0.25
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 1), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 1), 'Product Sales']*7
        #Demo for early months
        df.loc[(df['STORE'] == 44.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 44.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 38.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 38.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 22.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 22.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 11.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 11.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 48.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 48.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 7.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 7.0) & (df['MONTH'] == 4), 'Product Sales']*5
        #Month 9:
        df.loc[(df['STORE'] == 6.0) & (df['MONTH'] == 9), 'Product Sales'] = df.loc[(df['STORE'] == 6.0) & (df['MONTH'] == 9), 'Product Sales']*0.2
    elif product == 1:
        df.loc[df['Cluster Id'] == 4, 'Product Sales'] = df.loc[df['Cluster Id'] == 4, 'Product Sales']*1.5
        df.loc[df['Cluster Id'] == 3, 'Product Sales'] = df.loc[df['Cluster Id'] == 3, 'Product Sales']*0.6
        df.loc[(df['STORE'] == 27.0) & (df['MONTH'] == 7), 'Product Sales'] = df.loc[(df['STORE'] == 31.0) & (df['MONTH'] == 8), 'Product Sales']*2
        df.loc[df['Cluster Id'] == 2, 'Product Sales'] = df.loc[df['Cluster Id'] == 2, 'Product Sales']*0.7
        df.loc[df['Cluster Id'] == 4, 'Product Sales'] = df.loc[df['Cluster Id'] == 4, 'Product Sales']*0.6
        df.loc[(df['STORE'] == 31.0) & (df['MONTH'] == 8), 'Product Sales'] = df.loc[(df['STORE'] == 31.0) & (df['MONTH'] == 8), 'Product Sales']*2
        #Demo for Month 8-10 Grunerlokka
        df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 8), 'Product Sales'] = df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 8), 'Product Sales']*5
        df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 9), 'Product Sales'] = df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 9), 'Product Sales']*5
        df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 10), 'Product Sales'] = df.loc[(df['STORE'] == 40.0) & (df['MONTH'] == 10), 'Product Sales']*0.25
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 8), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 8), 'Product Sales']*5
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 9), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 9), 'Product Sales']*5
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 10), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 10), 'Product Sales']*0.25
        df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 1), 'Product Sales'] = df.loc[(df['STORE'] == 45.0) & (df['MONTH'] == 1), 'Product Sales']*7
        #Demo for early months
        df.loc[(df['STORE'] == 47.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 47.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 33.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 33.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 21.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 21.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 16.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 16.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 49.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 49.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 37.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 37.0) & (df['MONTH'] == 4), 'Product Sales']*5
        df.loc[(df['STORE'] == 17.0) & (df['MONTH'] == 4), 'Product Sales'] = df.loc[(df['STORE'] == 17.0) & (df['MONTH'] == 4), 'Product Sales']*5
        #Month 9:
        df.loc[(df['STORE'] == 16.0) & (df['MONTH'] == 9), 'Product Sales'] = df.loc[(df['STORE'] == 16.0) & (df['MONTH'] == 9), 'Product Sales']*0.2
        
    return df

def createStorePositions(df):
    #Rema and Kiwi stores in Oslo
    lat = []
    long = []
    templist = np.array([[59.914665985544495, 10.747027729093002],[59.90463941306384, 10.756640765777925],
                [59.914665985544495, 10.751490924696716],[59.91501019177722, 10.756726596462613],
                [59.91522531886014, 10.72205099984914],[59.911094635256596, 10.765138003561917],
                [59.91879622476816, 10.764108035345677],[59.91987172355111, 10.746941898408314],
                [59.92202261655828, 10.728745793254712],[59.9210762408104, 10.717072820137304],
                [59.9151392681943, 10.774751040246842],[59.909244266545194, 10.779128405165869],
                [59.922409762494425, 10.770373675327814],[59.92413035645291, 10.77209028902155],
                [59.92417337015878, 10.75114760195797],[59.92628097344073, 10.745396946083954],
                [59.91595674051305, 10.787797304319238],[59.92610892921151, 10.72608504202942],
                [59.92765729515816, 10.722222661218515],[59.928818522197226, 10.752606723597646],
                [59.90752290073564, 10.795693727310425],[59.930581789238, 10.713982915488579],
                [59.92223769819152, 10.792088838553578],[59.930581789238, 10.780415865436172],
                [59.93244951511961, 10.789481096333281],[59.93520158967025, 10.791197710027015],
                [59.93491450601513, 10.76142595998992],[59.91775316643569, 10.806572900135183],
                [59.92833489125882, 10.759108531503376],[59.915480707857455, 10.753052797905337],
                [59.91259791128953, 10.751765337635035],[59.91534368741738, 10.719992567632666],
                [59.91422501534424, 10.747029234696045],[59.90316534392907, 10.768486905867748],
                [59.91869947741323, 10.741192748137342],[59.91444014751594, 10.774838376534571],
                [59.91715069341938, 10.712267807397888],[59.908071645038035, 10.76093380561531],
                [59.92523799084609, 10.765568662588398],[59.928033672418664, 10.766770292174012],
                [59.922485089585855, 10.75715725548909],[59.92885082715372, 10.732523848983975],
                [59.928076681064404, 10.718533447380025],[59.92399061076104, 10.739561965128294],
                [59.920076113622784, 10.751578260984449],[59.920076113622784, 10.764281202318095],
                [59.91590300930742, 10.788828778138523],[59.92270016822158, 10.705143860568882],
                [59.92386156876543, 10.771748471885846],[59.90970691764994, 10.787026333760101],
                ])
    for i in range(0, len(templist)):
        for j in range(0,2):
            if(j==0):
                lat.append(templist[i][j])
            else:
                long.append(templist[i][j])
    lat = templist[:,0]
    long = templist[:,1]
    lat = np.tile(lat, 12)
    long = np.tile(long,12)

    dfy = df.assign(Lat=lat, Long=long)
    return dfy
    #animation_fra

def runAllClustering(DfList):
    
    for i in DfList:
        clusterSpecificProduct(i)
    

def clusterSpecificProduct(inputList, product):
    inputDf = inputList[product]
    
    ClusterSet = inputDf[(inputDf['MONTH'] == 0)]

    #Init sklearn objects
    Sc = StandardScaler()

    if product == 0 : 
        kmeans = KMeans(
            init="random",
            n_clusters=3,
            n_init='auto',
            max_iter=3000
        )
    else:
        kmeans = KMeans(
            init="random",
            n_clusters=5,
            n_init='auto',
            max_iter=3000
        )

    #Run pca to find plottable data
    pca = PCA(n_components=3)

    #Fit pandas dataframe
    X = Sc.fit_transform(ClusterSet)

    #Fit PCA and kmeans
    kmeans.fit(X)
    pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2', 'PC3'])
    pca_data['cluster'] = pd.Categorical(kmeans.labels_)

    '''
    %matplotlib tk
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure= False)
    fig.add_axes(ax)

    sc = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], s = 40, alpha = 1, c= pca_data['cluster'])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

    plt.show()
    '''
    clusterStoreLabels = pd.DataFrame(kmeans.labels_, index = ClusterSet.STORE, columns = ['Cluster Id'])
    
    plottingArray = applyClusterLabels(inputDf, clusterStoreLabels, product)
    plottingArray = createStorePositions(plottingArray)
    plottingArray = plottingArray.dropna()
    
    fig = px.scatter_mapbox(plottingArray,
                        animation_frame= 'MONTH', animation_group= 'STORE', 
                        lat="Lat", lon="Long", size="Product Sales",
                        opacity = 1,color_continuous_scale='blackbody', 
                        color = 'Cluster Id', mapbox_style = "open-street-map",
                        height = 900)
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=list(
                    [
                    dict(
                        label="(All)",
                        method="update",
                        args=[{'data_frame ' : plottingArray}],
                ),
                dict(
                    label="Cluster 0",
                    method="update",
                    args=[{'data_frame ' : plottingArray[plottingArray['Cluster Id'] == 0]}]
                ),
                dict(
                    label="Cluster 1",
                    method="update",
                    args=[{'data_frame ' : plottingArray[plottingArray['Cluster Id'] == 1]}]
                ),
                dict(
                    label="Cluster 2",
                    method="update",
                    args=[{'data_frame ' : plottingArray[plottingArray['Cluster Id'] == 2]}]
                ),
                ]),
            ), dict(
                type="dropdown",
                direction="down"
        )
    ])
    

    #fig.show()
    return fig
