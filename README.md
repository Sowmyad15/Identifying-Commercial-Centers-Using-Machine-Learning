# Identify Commercial Centers Using Machine Learning

Even though several global data are available regarding geolocations, demography of the planet, they are of not in supervised structure from which insights cannot be drawn. A Commercial Centre contains a high concentration of business, civic and cultural activities, also known as downtown. It is important to get to know the commercial city centres if you want to start any business as it also helps in identifying customer needs and in developing your business too.
To identify commercial centre of any city, clustering of Point of Interest(POI) of the city data with the correct amenities of interest is needed. 
This web app provides the Commercial centre of the city using Machine Learning.

A city from user is taken, whose data is fetched from Open Street Map (OSM),after pre-processing the data, outliers are removed using Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and clusters are plotted on map using K-Means++.Along with identifying the commercial centre, it also forms clusters of top 5 amenities in the city too.

## Outline of the project

1.Fetch City details from OSM

2.Remove outliers using DBSCAN

3.Cluster using KMeans++

4.Plot cluster in Folium Map

5.Group the amenities and cluster the top 5 amenities

## Module Description

  The Streamlit app on “Identifying Commercial Centre Using Machine Learning” is divided into 5 different modules:
  
•	**app.py**- Comprises of Streamlit UI and function calls to cluster_model.py to identify the commercial centers.

•	**cluster_model.py**- It has the functions to get the city details, remove the outliers, form clusters and plot in map. It contains functions to group the various amenities, clusters and plot in the map.

•	**config.py**- It contains the configurations for few variables.

•	**convex_hull.py**- It uses Jarvis's algorithm for creating convex hull and defines a function apply_convex_hull() that returns the coordinates of the convex hull polygon.

•	**map_legend.py**- It contains draggable legend macro that can be added to the map.

Here, is the Streamlit app for [Identifying Commercial Centres Using Machine Learning](https://identifying-commercial-centres-using-ml.streamlit.app/)


