#!/usr/bin/env python
# coding: utf-8

#  ## Lab 2: Data wrangling 

# ## Import Libraries and Define Auxiliary Functions
# 

# We will import the following libraries.
# 

# In[ ]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np


# ### Data Analysis 
# 

# Load Space X dataset, from last section.
# 

# In[ ]:


df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)


# Identify and calculate the percentage of the missing values in each attribute
# 

# In[ ]:


df.isnull().sum()/df.count()*100


# Identify which columns are numerical and categorical:
# 

# In[ ]:


df.dtypes


# In[ ]:


# Apply value_counts() on column LaunchSite
df['LaunchSite'].value_counts()


# In[ ]:


# Apply value_counts on Orbit column
df['Orbit'].value_counts()


# ### TASK 3: Calculate the number and occurence of mission outcome of the orbits
# 

# Use the method <code>.value_counts()</code> on the column <code>Outcome</code> to determine the number of <code>landing_outcomes</code>.Then assign it to a variable landing_outcomes.
# 

# In[ ]:


landing_outcomes = df['Outcome'].value_counts()
landing_outcomes


# In[ ]:


for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)


# We create a set of outcomes where the second stage did not land successfully:
# 

# In[ ]:


bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes


# In[ ]:


# landing_class = 0 if bad_outcome
# landing_class = 1 otherwise
landing_class = []
for i in range(len(df)):
    if df['Outcome'][i] in bad_outcomes:
        landing_class.append(0)
    else:
        landing_class.append(1)


# In[ ]:


df['Class']=landing_class
df[['Class']].head(8)


# In[ ]:


df.head(5)


# We can use the following line of code to determine  the success rate:
# 

# In[ ]:


df["Class"].mean()


# In[ ]:


N = []
for i in range(len(df)):
    if 'None' in df['Outcome'][i]:
        N.append(df['Outcome'][i])
len(N)
