import pandas as pd
import numpy as np
import overpy
from sklearn.cluster import KMeans,DBSCAN
from convex_hull import *
import config
from maplegend import *

import plotly.express as px
import folium
from folium import plugins 


api = overpy.Overpass()

#Get City details
def fetch_city_data(city_name):
   
    res = api.query(f"""[out:json];
    area[name='{city_name}'][boundary=administrative]->.searchArea;
    (node[place=city](area.searchArea);
    node["amenity"](area.searchArea);
    way["amenity"](area.searchArea);
        relation["amenity"](area.searchArea);
        );
    (._;
    >;
    );
    out;
    """)
    return df_preprocess(res)

#Remove unnecessary amenity
def df_preprocess(res):
    tags = []
    for i in res.nodes:
        if len(i.tags) != 0:
            i_tags = i.tags
            i_tags['node_id'] = i.id
            i_tags['lat'] = float(i.lat)
            i_tags['lon'] = float(i.lon)
            tags.append(i.tags)
    df = pd.DataFrame(tags) #Convert JSON to DataFrame
    df = df[['node_id', 'lat', 'lon', 'name', 'amenity']]   #Select only the necessary columns
    df = df.dropna(subset=['node_id', 'lat', 'amenity'])
    remove_amenity =  [
    'arts_centre',
    "Ayurvedic Hospital",
    "baby_hatch",
    "bench",
    "bicycle_parking",
    "bicycle_rental",
    "bicycle_repair_station",
    "bureau_de_change",
    "car_rental",
    "car_wash",
    "charging_station",
    "fountain",
    "grave_yard",
    "House",
    "language_school",
    "loading_dock"
    "meditation_centre",
    "motorcycle_parking",
    "orphanage",
    "parking entrance"
    "payment_terminal",
    "photo_booth",
    "post_depot",
    "recycling",
    "shelter",
    "social_centre",
    "social_facility",
    "telephone",
    "training",
    "tuition",
    "vending_machine",
    "veterinary",
    "waste_basket",
    "waste_disposal",
    "waste_transfer_station",
    "water_point",
    "weighbridge",]
    #Removing unnecessary amenity
    for val in remove_amenity:
        df = df[df.amenity != val]
    df.name = np.where(df.name.isnull(), df.amenity, df.name)
    return df

#Call the cluster models
def cluster_models(data):
    db_return = outlier_dbscan(data)  #DBSCAN to remove outliers, returns df and no. of clusters
    df =db_return[0]  
    n_cluster=db_return[1]
    km_return=cluster_Kmeans(df,n_cluster) #Returns no.of clusters
    return clusters_convex(km_return)

#DBSCAN to remove outliers and get num_clusters
def outlier_dbscan(data):
    x=data.copy()
    coords = x[['lat', 'lon']].to_numpy() #converts lat,lon columns of df to numpy
    dbsc = (DBSCAN(eps=config.epsilon,min_samples=config.min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords)))
    cluster_labels = dbsc.labels_
    num_clusters = len(set(cluster_labels))     #Number of clusters formed by dbscan
    clusters = pd.Series([ coords[ cluster_labels == n ] for n in range (num_clusters) ])
    core_samples = np.zeros_like(cluster_labels, dtype='bool')
    core_samples[dbsc.core_sample_indices_] = True
    s = pd.Series(core_samples, name='bools')
    dbscan_return=[x[s.values],num_clusters] #Contains df after removal of outlier and num_clusters
    return dbscan_return
    
def cluster_Kmeans(data, num_clusters):
    x=data.copy()
    coords=x[['lat','lon']].to_numpy()
    kmeans = KMeans(num_clusters, init = 'k-means++', random_state = config.random_state)
    y_kmeans = kmeans.fit_predict(coords) #form clusters using kmeans
    km=[num_clusters,coords,y_kmeans,data]
    return km


def clusters_convex(km_return):
    num_clusters=km_return[0] #no.of clusters
    coords=km_return[1] #Coords (lat,lon)
    y_kmeans=km_return[2] #KMeans Clusters
    df=km_return[3] #dataframe 
    most_significant = []
    least_significant = []
    for i in range(num_clusters):
        if len(coords[y_kmeans == i]) > 5:
            if len(coords[y_kmeans == i]) > 45: #If len of coords in the cluster >45 then the cluster is appended to most_significant else to least_significant
                most_significant.append(apply_convex_hull(coords[y_kmeans == i])) # apply convex hull to the clutser, results in a polygon
            else:
                least_significant.append(apply_convex_hull(coords[y_kmeans == i]))
    
    return most_significant,least_significant,coords



def mapplot(most_significant,least_significant,coords):
    #Creat a Map
    map_osm = folium.Map(location=coords[0])
    #Add Plugins
    # add tiles to map, Create a tile layer to append on a Map
    folium.raster_layers.TileLayer('Open Street Map').add_to(map_osm)
    folium.raster_layers.TileLayer('Stamen Terrain').add_to(map_osm)
    folium.raster_layers.TileLayer('Stamen Toner').add_to(map_osm)
    folium.raster_layers.TileLayer('Stamen Watercolor').add_to(map_osm)
    folium.raster_layers.TileLayer('CartoDB Positron').add_to(map_osm)
    folium.raster_layers.TileLayer('CartoDB Dark_Matter').add_to(map_osm)
    # add layer control to show different maps
    folium.LayerControl().add_to(map_osm)
    minimap = plugins.MiniMap(toggle_display=True,position='bottomleft')
    # add minimap to map
    map_osm.add_child(minimap)
    # add full screen button to map
    plugins.Fullscreen(position='topright').add_to(map_osm)
    # create a polygon with the coordinates
    for cords in coords:
        folium.CircleMarker(location=[cords[0], cords[1]],radius=1,color='blue').add_to(map_osm)

    #Color the polygon that are least significant with yellow
    for i in range(len(least_significant)):
        folium.Polygon(least_significant[i],
               color="blue",
               weight=2,
               fill=True,
               fill_color="yellow",
               fill_opacity=0.4).add_to(map_osm)

     #Color the polygon that are most significant with yellow
    for i in range(len(most_significant)):
        folium.Polygon(most_significant[i],
               color="black",
               weight=2,
               fill=True,
               fill_color="red",
               fill_opacity=0.4).add_to(map_osm)

    #Legend
    
    macro=add_map_legend()

    map_osm.add_child(macro)

    return map_osm

def amenity_df(city_data):
    #Group the amenity
    food_list = ['restaurant', 'fast_food', 'cafe', 'bar', 'ice_cream', 'fast_food','bar', 'food_court', 'club', 'drinking_water']
    market_list = ['marketplace', 'internet_cafe']
    bank_list = ['atm', 'bank', ]
    toilets_list = ['toilets']
    education_list = ['school', 'college', 'university']
    hospital_list = ['pharmacy', 'hospital', 'clinic', 'dentist', 'nursing_home']
    parking_list = ['parking']
    entertainment_list = ['cinema', 'theatre', 'nightclub', 'cafe','coworking_space', 'studio', 'internet_cafe', 
                 'swimming_pool', 'library','pub']
    worship_list = ['place_of_worship']
    fuel_list = ['fuel', 'fire_station']
    others_list = ['post_box', 'community_centre', 'post_office', 'embassy', 'police', 'bus_station', 'public_building',
                'taxi']
    
    amenity_list = ['food_list', 'market_list', 'bank_list', 'toilets_list', 'education_list', 
    'hospital_list', 'parking_list', 'entertainment_list', 'worship_list', 'fuel_list', 'others_list']
    food,market,bank,toilets,education,hospital,parking,entertainment,worship,fuel,others = [],[],[],[],[],[],[],[],[],[],[]

    for value in city_data.values:
        if value[4] in food_list:     food.append([value[1], value[2]])
        elif value[4] in market_list:     market.append([value[1], value[2]])    
        elif value[4] in bank_list:     bank.append([value[1], value[2]])    
        elif value[4] in toilets_list:     toilets.append([value[1], value[2]])    
        elif value[4] in education_list:     education.append([value[1], value[2]])    
        elif value[4] in hospital_list:     hospital.append([value[1], value[2]])    
        elif value[4] in parking_list:     parking.append([value[1], value[2]])    
        elif value[4] in entertainment_list:     entertainment.append([value[1], value[2]])    
        elif value[4] in worship_list:     worship.append([value[1], value[2]])    
        elif value[4] in fuel_list:     fuel.append([value[1], value[2]])    
        elif value[4] in others_list:     others.append([value[1], value[2]])   
    
    amenities_list = [food,market,bank,toilets,education,hospital,parking,entertainment,worship,fuel,others]
    amenities_str = ['Food','Market','Bank','Toilets','Education','Hospital','Parking','Entertainment','Worship','Fuel','Others']
    x=[]

    for i,item in enumerate(amenities_list):
        x.append(len(item))

    #Create df  that has amenity along with  their count in ascending
    df=pd.DataFrame(list(zip(amenities_str,amenities_list,x)),columns=['Amenity','lat_lon','Count']).sort_values(by=['Count'],ascending=False)

    return df

def barplot(df):
    dx_plot=df[['Amenity','Count']]

    #Create bar plot for Amenity-Count
    fig = px.bar(dx_plot, x='Amenity', y='Count',color='Count',width=725,height=500)

    return fig

def top5(df,ilocation):

    #'df' Contains 'Amenity','lat_lon','Count' column

    amenity_name = df.iloc[ilocation,0] #Contain Amenity name
    amenity_array = df.iloc[ilocation,1]    #Contain Amenity coordinates

    amenities_df = pd.DataFrame(amenity_array, columns = ['lat', 'lon'])
    coords_amenity=amenities_df[['lat','lon']].to_numpy()

    # Fitting K-Means to the dataset
    if len(amenity_array) < 60: 
        n_clusters = 5
    else: 
        n_clusters = 20

    kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(amenity_array)


    polygon = []
    amenity_array = np.array(amenity_array)
    for i in range(n_clusters):
        polygon.append(apply_convex_hull(amenity_array[y_kmeans == i]))

    polygon = [i for i in polygon if i is not None]

    #Create map for the amenity
    amenity_map_osm = folium.Map(location=coords_amenity[0])
    #Add Plugins
    # add tiles to map, Create a tile layer to append on a Map
    folium.raster_layers.TileLayer('Open Street Map').add_to(amenity_map_osm)
    folium.raster_layers.TileLayer('Stamen Terrain').add_to(amenity_map_osm)
    folium.raster_layers.TileLayer('Stamen Toner').add_to(amenity_map_osm)
    folium.raster_layers.TileLayer('Stamen Watercolor').add_to(amenity_map_osm)
    folium.raster_layers.TileLayer('CartoDB Positron').add_to(amenity_map_osm)
    folium.raster_layers.TileLayer('CartoDB Dark_Matter').add_to(amenity_map_osm)
    # add layer control to show different maps
    folium.LayerControl().add_to(amenity_map_osm)
    minimap = plugins.MiniMap(toggle_display=True)
    # add minimap to map
    amenity_map_osm.add_child(minimap)
    # add full screen button to map
    plugins.Fullscreen(position='topright').add_to(amenity_map_osm)
    # create a polygon with the coordinates
    for cords in coords_amenity:
        folium.CircleMarker(location=[cords[0], cords[1]],radius=1,weight=1).add_to(amenity_map_osm)

    for i in range(len(polygon)):
        folium.Polygon(polygon[i],
               color="black",
               weight=2,
               fill=True,
               fill_color="yellow",
               fill_opacity=0.4).add_to(amenity_map_osm)

    return amenity_map_osm


