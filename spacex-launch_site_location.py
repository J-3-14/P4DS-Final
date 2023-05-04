#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import folium
import pandas as pd


# In[ ]:


# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
#from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon


# In[ ]:


# Download and read the `spacex_launch_geo.csv`
#from js import fetch
import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
#resp = await fetch(URL)
#spacex_csv_file = io.BytesIO((await resp.arrayBuffer()).to_py())
#spacex_df=pd.read_csv(spacex_csv_file)
spacex_df=pd.read_csv(URL)


# In[ ]:


# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites = launch_sites[['Launch Site', 'Lat', 'Long']]
launch_sites


# In[ ]:


# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)


# In[ ]:


# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)


# In[ ]:


# Initial the map
# site_map = folium.Map(location=nasa_coordinate, zoom_start=5)
# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label
site_map = folium.Map(location=[30,-100], zoom_start=5)

for label, lat, lng in zip(launch_sites['Launch Site'], launch_sites['Lat'], launch_sites['Long']):
    circle = folium.Circle([lat,lng],radius=5,color='red',fill=True,fill_color='blue',fill_opacity=0.6).add_to(site_map)
    marker = folium.map.Marker([lat,lng],icon=DivIcon(icon_size=(20,20),icon_anchor=(0,0),html='<div style="font-size:12; color:#d35400;"><b>%s</b></div>' % f'{label}',)).add_to(site_map)


site_map


# In[ ]:


spacex_df.tail(10)


# In[ ]:


marker_cluster = MarkerCluster()


# In[ ]:


# Apply a function to check the value of `class` column
# If class=1, marker_color value will be green
# If class=0, marker_color value will be red
colors = []
for i in range(len(spacex_df)):
    if spacex_df['class'][i] == 0:
        colors.append('red')
    else:
        colors.append('green')
spacex_df['marker_color']=colors


# In[ ]:


l = []
for i in range(len(spacex_df)):
    if 'CAFS LC-40' in spacex_df['Launch Site'][i]:
        l.append(spacex_df['class'][i])
l    


# In[ ]:


site_map.add_child(marker_cluster)
for lat, lng, color, cls in zip(spacex_df['Lat'],spacex_df['Long'],spacex_df['marker_color'],spacex_df['class']):
    marker = folium.Marker([lat,lng],icon=DivIcon(icon_size=(20,20))).add_child(folium.Popup(cls))
    marker_cluster.add_child(marker)
site_map


# In[ ]:


from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


# In[ ]:


coordinates=pd.DataFrame([(28.563197,-80.576820),(28.563197,-80.5768)],columns=['Lat','Long'])


# In[ ]:


# find coordinate of the closet coastline
# e.g.,: Lat: 28.56367  Lon: -80.57163

launch_site_lat = 28.563197
launch_site_lon = -80.576820
coastline_lat = 28.563197
coastline_lon = -80.58

distance = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)
distance


# In[ ]:


# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon property 
# for example
distance_marker = folium.Marker(
   [coastline_lat,coastline_lon],
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % f"{distance} KM",
       )
   )
site_map.add_child(distance_marker)


# In[ ]:


# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate
lines=folium.PolyLine(locations=[28.563197,-80.5768], weight=1)
site_map.add_child(lines)

