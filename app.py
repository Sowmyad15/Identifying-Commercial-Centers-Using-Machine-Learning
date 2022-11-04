import pandas as pd
import streamlit as st
from cluster_model import *
from streamlit_folium import folium_static

#reducing heading space

#SET PAGE WIDE
st.set_page_config(page_title='Identifying Commercial Centre Using Machine Learning',layout="centered")


st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

#About and Contact
st.sidebar.title("About")
st.sidebar.info(
    """
    [GitHub repository](https://github.com/Sowmyad15/Identifying-Commercial-Centers-Using-Machine-Learning)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Sowmya D
    [GitHub](https://github.com/Sowmyad15) | [LinkedIn](https://www.linkedin.com/in/sowmya-d-4b01711b2/)
    """
)
#Title of the page with CSS

st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Verdana'; color: 	#000000;} 
        </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Identifying Commercial Centers</p>', unsafe_allow_html=True) 

with st.expander("About this project"):
    st.info("""Even though several global data are available regarding geolocations, demography of the planet, they are of not in supervised structure from which insights cannot be drawn. A Commercial Centre contains a high concentration of business, civic and cultural activities, also known as downtown. It is important to get to know the commercial city centres if you want to start any business as it also helps in identifying customer needs and in developing your business too.
To identify commercial centre of any city, clustering of Point of Interest(POI) of the city data with the correct amenities of interest is needed. 
This web app provides the Commercial centre of the city using Machine Learning.
 """)

#Get city name from user
city = st.text_input('Enter City Name:',help="City name is case sensitive, Kindly provide the exact name as in OSM")


if city:
        with st.spinner("Fetching City Data"):
            try:
                #call fetch_city_data(), that fetches the city data and returns it in dataframe

                city_data=fetch_city_data(city)

                st.header("Point of Interest in "+city)
                st.markdown("The following data represents the Point of Interest Data of "+city)

                #Display the city_data df to the user
                st.dataframe(city_data,width=1000,height=500,use_container_width=True)

                #Cluster of amenities
                x=cluster_models(city_data)

                with st.expander("How it works?"):
                    st.success("""A city from user is taken, whose data is fetched from Open Street Map (OSM), 
                    after pre-processing the data, outliers are removed using Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and 
                    clusters are plotted on map using K-Means.Along with identifying the commercial centre, it also forms clusters of top 5 amenities in the city too.""")

                st.header("Commercial Centers in "+city)
                
                #Plot the clutsers in map
                folium_static(mapplot(x[0],x[1],x[2]),height=500)

                #get the amenities in the city
                city_amenity=amenity_df(city_data)

                #Get the top5 amenities,and cluster of the same
                top5name=list(city_amenity.iloc[:,0])

                #Plot the amenity using barplot
                barplt=barplot(city_amenity)

                #Tabs containing the amenity bar plot and cluster of top 5 amenities 
                tab1, tab2,tab3,tab4,tab5,tab6= st.tabs(["📈 Chart",top5name[0],top5name[1],top5name[2],top5name[3],top5name[4]])

                with tab1:
                    st.subheader("Top Amenities")
                    st.plotly_chart(barplt)

                with tab2:
                    st.header(top5name[0])
                    folium_static(top5(dx,0),width=725,height=500)

                with tab3:
                    st.header(top5name[1])
                    folium_static(top5(dx,1),width=725,height=500)

                with tab4:
                    st.header(top5name[2])
                    folium_static(top5(dx,2),width=725,height=500)

                with tab5:
                    st.header(top5name[3])
                    folium_static(top5(dx,3),width=725,height=500)

                with tab6:
                    st.header(top5name[4])
                    folium_static(top5(dx,4),width=725,height=500)

            except KeyError:
                st.error("Oops Couldnt find city details")

        
