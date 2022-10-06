import pandas as pd
import streamlit as st
from cluster_model import *
import config
from streamlit_folium import folium_static


st.header("Identifying Commercial Centers")
st.markdown("This project focuses on identifying commercial centers")

city = st.text_input('Enter City Name:')
if city:
        with st.spinner("Fetching City Dta"):
            try:
                df=fetch_city_data(city)

                st.header("Commercial Centers In "+city)
                st.markdown("The following data represents the commercial centers in "+city)
                st.dataframe(df,width=725,height=500)
                x=cluster_models(df)
                st.markdown("Point Of Interest of "+city)
                folium_static(mapplot(x[0],x[1],x[2]),width=725,height=500)

                dx=amenity_df(df)

                top5name=list(dx.iloc[:,0])

                barplt=barplot(dx)

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

        
