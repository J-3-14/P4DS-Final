#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
#df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")


# In[ ]:


df.head()


# In[ ]:


df['Launch_Site'].unique()


# In[ ]:


KSC = []
for i in range(len(df)):
    if 'KSC' in df['Launch_Site'][i]:
        KSC.append([df['Booster_Version'][i], df['Mission_Outcome'][i]])
KSC[0:5]


# In[ ]:


df.columns


# In[ ]:


CRS = []
for i in range(len(df)):
    if '(CRS)' in df['Customer'][i]:
        CRS.append(int(df['PAYLOAD_MASS__KG_'][i]))
len(CRS)


# In[ ]:


import numpy as np


# In[ ]:


CR = np.array(CRS)
CR.sum()    


# In[ ]:


F = []
for i in range(len(df)):
    if 'F9 v1.1' in df['Booster_Version'][i]:
        F.append(int(df['PAYLOAD_MASS__KG_'][i]))
FA = np.array(F)
FA.mean()


# In[ ]:


D = []
for i in range(len(df)):
    if 'Success (drone ship)' in df['Landing _Outcome'][i]:
        D.append(df['Date'][i])
D[0]


# In[ ]:


GP = []
for i in range(len(df)):
    if 'Success (ground pad)' in df['Landing _Outcome'][i]:
        GP.append([df['Booster_Version'][i], df['PAYLOAD_MASS__KG_'][i]])
for i in range(len(GP)):
    if 4000 < int(GP[i][1]) < 6000:
        print(GP[i])


# In[ ]:


df['Mission_Outcome'].value_counts()


# In[ ]:


for i in range(len(df)):
    if df['PAYLOAD_MASS__KG_'][i] == df['PAYLOAD_MASS__KG_'].max():
        print(df['Booster_Version'][i])


# In[ ]:


t = []
for i in range(len(df)):
    if '2017' in df['Date'][i] and 'Success (ground pad)' in df['Landing _Outcome'][i]:
        t.append([df['Date'][i], df['Booster_Version'][i], df['Launch_Site'][i]])
t


# In[ ]:


Z = []
DF = df[:][0:30]
for i in range(len(DF)):
    if 'Success' in DF['Landing _Outcome'][i]:
        Z.append(DF['Date'][i])
print(Z)

