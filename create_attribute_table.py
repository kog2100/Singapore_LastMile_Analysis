import pandas as pd
import geopandas as gpd
import numpy as np

import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#load in the OD matrix for POI and bus stops extracted from GIS
fp_dest100 = "OD_allPOI_100.csv"
fp_dest300 = "OD_allPOI_300.csv"
fp_dest500 = "OD_allPOI_500.csv"
fp_dest400 = "OD_allPOI_400.csv"
fp_dest200 = "OD_allPOI_200.csv"
fp_dest600 = "OD_allPOI_600.csv"
fp_trips = "trips_under_timeBand.csv"

df_DestBus100 = pd.read_csv(fp_dest100)
df_DestBus300 = pd.read_csv(fp_dest300)
df_DestBus500 = pd.read_csv(fp_dest500)
df_DestBus200 = pd.read_csv(fp_dest200)
df_DestBus400 = pd.read_csv(fp_dest400)
df_DestBus600 = pd.read_csv(fp_dest600)
df_trips = pd.read_csv(fp_trips)

df_trips.describe(include='object')

dest600_summary.describe()

df_DestBus600.var()

poi_count = {'poi_100': len(df_DestBus100.poi.unique()), 'poi_200':len(df_DestBus200.poi.unique()), 'poi_300':len(df_DestBus300.poi.unique()), 'poi_400': len(df_DestBus400.poi.unique()), 'poi_500': len(df_DestBus500.poi.unique()), 'poi_600':len(df_DestBus600.poi.unique())}

poi_count = pd.Series(poi_count)

poi_count

ax = poi_count.plot.bar(figsize=(6,5))
plt.xlabel("Distance buffers")
plt.ylabel("count of POIs")

round ((np.log(0.001)/-600), 4)

x = np.linspace(0,100,10)
x

#plot decay function
def decay_graph():
    beta = 0.0115
    x = np.linspace(1,600,10)
    y= np.exp(-beta * x)
    plt.plot(x,y)
    plt.xlabel("Distance in meters")
    plt.ylabel("Accessibility Score")

decay_graph()

#plot a sample negative binomial distribution
from scipy.stats import nbinom
data_nb = nbinom.rvs(3,0.2, size =10000)
sns.set_style('whitegrid')
ax = sns.distplot(data_nb, kde=False,)
ax.set(xlabel='Negative Binomial', ylabel='Frequency')

def dest_summary_table (df_DestBus, df_trips, out_name):
    
    '''
    Generate the accessibility score table by first calculating the access scores for the bus stops and 
    their POI and then merging with the trips data
    Requires three inputs
    1. A dataframe containing the bus stops (dest_busid) and their distance (Total_Leng) to POIs (poi). 
    2. A trip table dataframe containing the exit bus stops (dest_busid), origin bus stops (orig_busid) and 
    the person trip ID (Card_Number)
    3. An appended string name that is used to identify the different distance buffer. E.g '100' to generate
    a table for 100m buffer 
    
    The function returns a dataframe containing access scores per bus stop, total number of destination trips,
    origin trips and pois per bus stop and average distance per bus stop
    
    '''

    
    #rename to show that the length is the raw length from GIS in meters
    df_DestBus.rename(columns={'Total_Leng':'raw_length'}, inplace = True)

    #there was infinity in the value and because of these two zero POI
    #print(df_DestBus[df_DestBus['raw_length']==0])

    df_DestBus = df_DestBus[df_DestBus['raw_length']!=0].copy()

    #df_DestBus['length'] = 1/((df_DestBus['raw_length'])**2)
    
    # set a beta value (b) based on f(x) = Ke^-bx K is 1 f(0) = k, for beta value use f(600) as the max dist
    beta = round ((np.log(0.001)/-600), 4)
    
    #apply distance decay for an accessibility score

    df_DestBus['length'] = df_DestBus['raw_length'].map(lambda x: math.exp(-beta * x))
    
    #get average length per bus stop
    dest_dist = df_DestBus.groupby('dest_busid')['raw_length'].mean().reset_index().rename(columns={'raw_length':'avg_distance'})

    #using sum
    dest_access = df_DestBus.groupby('dest_busid')['length'].sum().reset_index().rename(columns={'length':'access_score'})

    #1 - group POIs by dest
    dest_poi = df_DestBus.groupby('dest_busid')['poi'].count().reset_index()

    #number of trips per dest
    dest_trips = df_trips.groupby(['dest_busid'])['Card_Number'].count().reset_index().rename(columns={'Card_Number':'TripsPerdest'})
    
    #number of trips per orig
    orig_trips = df_trips.groupby(['orig_busid'])['Card_Number'].count().reset_index().rename(columns={'orig_busid':'dest_busid','Card_Number':'TripsPerOrig'})

    
    # merge the aggregates which reflects the distinct trips per bus stop
    from functools import reduce
    dfs = [dest_trips,orig_trips,dest_poi, dest_dist, dest_access]
    dest_summary = reduce(lambda left,right: pd.merge(left,right,on='dest_busid', how='left'), dfs)

    dest_summary.dest_busid = dest_summary.dest_busid.astype(str)
    poi_name = 'poi'+ '_'+ out_name
    avg_dist_name = 'avg_distance'+ '_'+ out_name
    access_name = 'access_score'+ '_'+ out_name
    dest_summary.rename(columns={'poi':poi_name, 'avg_distance':avg_dist_name,
       'access_score':access_name}, inplace=True)
    
    return dest_summary

dest200_summary = dest_summary_table (df_DestBus200, df_trips, '200')
dest400_summary = dest_summary_table (df_DestBus400, df_trips, '400')
dest600_summary = dest_summary_table (df_DestBus600, df_trips, '600')
dest100_summary = dest_summary_table (df_DestBus100, df_trips, '100')
dest300_summary = dest_summary_table (df_DestBus300, df_trips, '300')
dest500_summary = dest_summary_table (df_DestBus500, df_trips, '500')

def csv_save (name, file):
    output = name
    file.to_csv(output, index=False)

#output file for regression analysis
csv_save('dest100_summary_single.csv',dest100_summary)
csv_save('dest200_summary_single.csv',dest200_summary)
csv_save('dest300_summary_single.csv',dest300_summary)
csv_save('dest400_summary_single.csv',dest400_summary)
csv_save('dest500_summary_single.csv',dest500_summary)
csv_save('dest600_summary_single.csv',dest600_summary)

#merge all the distance buffer access dataframes for better descriptive analysis
from functools import reduce
dfs = [dest100_summary,dest200_summary,dest300_summary,dest400_summary,dest500_summary,dest600_summary]
dest_summary = reduce(lambda left,right: pd.merge(left,right,on='dest_busid', how='left'), dfs)

dest_summary.columns

dest_summary.columns = ['dest_busid', 'TripsPerdest', 'TripsPerOrig', 'poi_100',
       'avg_distance_100', 'access_score_100', 'TripsPerdest_y',
       'TripsPerOrig_y', 'poi_200', 'avg_distance_200', 'access_score_200',
       'TripsPerdest_x', 'TripsPerOrig_x', 'poi_300', 'avg_distance_300',
       'access_score_300', 'TripsPerdest_y1', 'TripsPerOrig_y2', 'poi_400',
       'avg_distance_400', 'access_score_400', 'TripsPerdest_x2',
       'TripsPerOrig_x1', 'poi_500', 'avg_distance_500', 'access_score_500',
       'TripsPerdest_y3', 'TripsPerOrig_y4', 'poi_600', 'avg_distance_600',
       'access_score_600']

dest_summary = dest_summary[['dest_busid', 'TripsPerdest', 'TripsPerOrig', 'poi_100',
       'access_score_100', 'poi_200',
       'access_score_200', 'poi_300',
       'access_score_300', 'poi_400',
       'access_score_400', 'poi_500',
       'access_score_500', 'poi_600',
       'access_score_600']].copy()



#know how many poi per distance for plotting
poi_only = dest_summary[['TripsPerdest','poi_100','poi_200','poi_300','poi_400','poi_500','poi_600']].copy()
poi_only.rename(columns={'TripsPerdest':'Total_BusStops','poi_100':'100m','poi_200':'200m','poi_300':'300m','poi_400':'400m','poi_500':'500m','poi_600':'600m'}, inplace=True)
poi_only= poi_only.count()

#know how check access sum per distance for plotting
access_only = dest_summary[['access_score_100','access_score_200','access_score_300','access_score_400','access_score_500','access_score_600']].copy()
access_only.rename(columns={'access_score_100':'100m','access_score_200':'200m','access_score_300':'300m','access_score_400':'400m','access_score_500':'500m','access_score_600':'600m'}, inplace=True)
access_only = access_only.sum()

print(poi_only)

ax = access_only.plot.bar()
plt.xlabel("Distance buffers")
plt.ylabel("Sum of access scores")

ax = poi_only.plot.bar()
plt.xlabel("Distance buffers")
plt.ylabel("number of bus stops with at least 1 poi within buffer")

#plot all histogram of the data 
dest_summary.hist(figsize=(40,40), bins=50)

#plot a scatter matrix to view the data
from pandas.plotting import scatter_matrix
scatter_matrix(dest_summary[['TripsPerdest', 'TripsPerOrig',
       'access_score_100', 'access_score_200',
       'access_score_300','access_score_400',
       'access_score_500', 'access_score_600']], alpha=0.2, figsize=(50,50), diagonal = 'kde')

sns.set_style('whitegrid')
sns.kdeplot(np.array(dest_summary['access_score_600']), bw=0.5)
