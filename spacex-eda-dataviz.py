#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# andas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns


# ## Exploratory Data Analysis 
# 

# In[ ]:


df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

# df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')

df.head(5)


# In[ ]:


# This plot takes a while to run on slower machines
plt.figure(figsize=[12,8])
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 3)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()


# In[ ]:


# Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 3) # aspect changed from 5 to 3
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()


# In[ ]:


# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df, aspect = 3) #aspect changed from 5 to 3
plt.xlabel("Payload",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()


# In[ ]:


# HINT use groupby method on Orbit column and get the mean of Class column
df.groupby('Orbit')['Class'].mean().plot(kind='bar')


# In[ ]:


# Plot a scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value
sns.catplot(x='Orbit', y='FlightNumber', data=df, hue='Class', aspect=3) # added aspect=3
plt.xlabel("Orbit",fontsize=20)
plt.ylabel("Flight Number",fontsize=20)
plt.show()


# In[ ]:


# Plot a scatter point chart with x axis to be Payload and y axis to be the Orbit, and hue to be the class value
sns.catplot(x='PayloadMass', y='Orbit', data=df, hue='Class', aspect=3) #added aspect=3
plt.xlabel("Payload Mass",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.show()


# In[ ]:


# A function to Extract years from the date 
year=[]
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year
Extract_year(df['Date'])   

df['Date'] = year
group = df.groupby('Date')['Class'].mean()

plt.figure(figsize=[12,8])
plt.plot(df['Date'].unique(), group)
plt.xlabel("Year",fontsize=20)
plt.ylabel("Success Rate",fontsize=20)
plt.show()


# In[ ]:


features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()


# In[ ]:


# HINT: Use get_dummies() function on the categorical columns
features_one_hot = features.copy()

features_one_hot['GridFins'] = pd.get_dummies(features['GridFins'])
features_one_hot['Reused'] = pd.get_dummies(features['Reused'])
features_one_hot['Legs'] = pd.get_dummies(features['Legs'])
features_one_hot['LandingPad'] = pd.get_dummies(features['LandingPad'])
features_one_hot['LaunchSite'] = pd.get_dummies(features['LaunchSite'])
features_one_hot['Serial'] = pd.get_dummies(features['Serial'])
features_one_hot['Orbit'] = pd.get_dummies(features['Orbit'])

features_one_hot.head()


# In[ ]:


# HINT: use astype function
features_one_hot = features_one_hot.astype('float64')
features_one_hot.head()


# Copyright © 2020 IBM Corporation. All rights reserved.
# 
