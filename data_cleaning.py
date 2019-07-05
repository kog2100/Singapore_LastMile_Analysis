import pandas as pd
import geopandas as gpd

#load in the shapefiles for the raw OD matrix from GIS
    all_poi = gpd.read_file('all_POI.shp')

    OD100 = gpd.read_file('ODMatrix_100.shp')
    OD200 = gpd.read_file('ODMatrix_200.shp')
    OD300 = gpd.read_file('ODMatrix_300.shp')
    OD400 = gpd.read_file('ODmatrix_400.shp')
    OD500 = gpd.read_file('ODMatrix_500.shp')
    OD600 = gpd.read_file('ODMatrix_600.shp')

#load in the trips
fp_trips = "trips_under_timeBand.csv"
df_trips = pd.read_csv(fp_trips)

def clean_OD(OD, all_poi):
    '''
    Function to split the Name column in the OD Matrix to be able to identify the origin and the destination separately
    This function also classifies POIs into groups
    '''

    #split name column - checked for white spaces [ -] means eith ' ' or '-'
    OD[['dest_busid','space','space2','poi']]= OD.Name.str.split('[ -]', expand = True)

    #drop empty columns
    OD.drop(['space','space2'], axis=1, inplace=True)


    #let's group the poi based on the trade code
    df_poi_group = all_poi[['TRADE_CODE','POSTAL_CD']].copy()

    df_poi_group.rename(columns={'POSTAL_CD':'poi'}, inplace=True)

    #remove duplicates so it doesn't affect the merge. However some locations with same postal codes are lost
    df_poi_group.poi.drop_duplicates(inplace = True)

    OD_poiGr = OD.merge(df_poi_group, on='poi')

    OD_poiGr['poi_group'] = OD_poiGr['TRADE_CODE'].copy()

    OD_poiGr['poi_group'].unique()

    OD_poiGr['poi_group'].replace(['9CCARE', '9CHNTE', '9SCTRE', '9PINT', '9MOS', '9CHU', '9SPT',
           '9CLNI', '9HOSPI', '9CC', '9LIB', '9INDTE', '9HOSP', '9RCLUB',
           '9SYNA', '9POLY', '9SWC', '9SHTEM', '9NPC'], ['child_care','culture','daily_needs','culture','culture','culture','sports','health_care','health_care','social','social','culture','health_care','social','culture','health_care','sports','culture','health_care'], inplace=True)

    #create OD tables for different groups of poi

    #first select relevant columns
    OD_poiGr = OD_poiGr[['Name', 'Total_Leng', 'geometry', 'dest_busid', 'poi', 'TRADE_CODE',
           'poi_group']].copy()
    return OD_poiGr
    


OD_poi_100 = clean_OD(OD100,all_poi)
OD_poi_200 = clean_OD(OD200,all_poi)
OD_poi_300 = clean_OD(OD300,all_poi)
OD_poi_400 = clean_OD(OD400,all_poi)
OD_poi_500 = clean_OD(OD500,all_poi)
OD_poi_600 = clean_OD(OD600,all_poi)

OD600.head()

#save the file as csv to be used
def csv_save (name, file):
    output = name
    file.to_csv(output, index=False)

csv_save('OD_allPOI_100.csv',OD_poi_100)
csv_save('OD_allPOI_200.csv',OD_poi_200)
csv_save('OD_allPOI_300.csv',OD_poi_300)
csv_save('OD_allPOI_400.csv',OD_poi_400)
csv_save('OD_allPOI_500.csv',OD_poi_500)
csv_save('OD_allPOI_600.csv',OD_poi_600)


def group_activities(filename, OD_poiGr):
    ''' 
    function to split the dataframe based on activity type of POIs
    returns nothing but saves the files to csv after splitting
    takes in the name of the file and appends a name to distinguish the different categories
    
    '''
   
    
    OD_child_care = OD_poiGr.loc[OD_poiGr['poi_group']=='child_care'].reset_index()

    OD_culture = OD_poiGr.loc[OD_poiGr['poi_group']=='culture'].reset_index()

    OD_shopping = OD_poiGr.loc[OD_poiGr['poi_group']=='daily_needs'].reset_index()

    OD_sport = OD_poiGr.loc[OD_poiGr['poi_group']=='sports'].reset_index()

    OD_health_care = OD_poiGr.loc[OD_poiGr['poi_group']=='health_care'].reset_index()

    OD_social = OD_poiGr.loc[OD_poiGr['poi_group']=='social'].reset_index()


    child_care_name = filename + '_child_care.csv'
    culture_name = filename + '_culture.csv'
    shopping_name = filename + '_shopping.csv'
    sport_name = filename + '_sport.csv'
    health_care_name = filename + '_health_care.csv'
    social_name = filename + '_social.csv'

    csv_save(child_care_name, OD_child_care)
    csv_save(culture_name, OD_culture)
    csv_save(shopping_name, OD_shopping)
    csv_save(sport_name, OD_sport)
    csv_save(health_care_name, OD_health_care)
    csv_save(social_name, OD_social)


group_activities('OD100_poi',OD_poi_100)
group_activities('OD200_poi',OD_poi_200)
group_activities('OD300_poi',OD_poi_300)
group_activities('OD400_poi',OD_poi_400)
group_activities('OD500_poi',OD_poi_500)
group_activities('OD600_poi',OD_poi_600)











