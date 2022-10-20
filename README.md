# Identify Commercial Centers Using Machine Learning
Even though several global data are available regarding geolocations, demography of the planet, they are of not in supervised structure from which insights cannot be drawn. A Commercial Centre contains a high concentration of business, civic and cultural activities, also known as downtown. It is important to get to know the commercial city centres if you want to start any business as it also helps in identifying customer needs and in developing your business too.
To identify commercial centre of any city, clustering of Point of Interest(POI) of the city data with the correct amenities of interest is needed. 
This web app provides the Commercial centre of the city using Machine Learning.

A city from user is taken, whose data is fetched from Open Street Map (OSM),after pre-processing the data, outliers are removed using Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and clusters are plotted on map using K-Means.Along with identifying the commercial centre, it also forms clusters of top 5 amenities in the city too.
