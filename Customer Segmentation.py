#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np


# In[3]:


from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score


# In[4]:


df= pd.read_csv("D:\Data sets\Mall_Customers.csv")


# In[5]:


df.head(20)


# In[6]:


df.rename(columns = {'Annual Income (k$)':'Annual_Income_inThousandUSD','Spending Score (1-100)':'Spending_Score' }, inplace = True)


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


sns.distplot(df['Annual_Income_inThousandUSD'])


# In[10]:


sns.histplot(df['Annual_Income_inThousandUSD'])


# In[11]:


df.columns


# In[12]:


columns=['Age', 'Annual_Income_inThousandUSD','Spending_Score']


# In[13]:


for i in columns:
 plt.figure()
 sns.distplot(df[i])


# In[14]:


sns.kdeplot(df['Annual_Income_inThousandUSD'],shade=True,hue=df['Gender']);


# In[15]:


for i in columns:
 plt.figure()
 sns.kdeplot(df[i],shade=True,hue=df['Gender']);


# In[16]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[17]:


#sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Age');
sns.scatterplot(data=df,x='Annual_Income_inThousandUSD',y='Spending_Score');


# In[18]:


x=df.copy()
y=x.drop('CustomerID',axis=1)
sns.pairplot(y,hue='Gender');


# In[19]:


df.groupby(['Gender'])['Age', 'Annual_Income_inThousandUSD','Spending_Score'].mean()


# In[19]:


#df.corr()
sns.heatmap(y.corr(),annot=True);


# # Clusterin(Univariate)

# In[20]:


df.head()


# In[21]:


clustering1= KMeans(5)


# In[22]:


#type(clustering1)
type('Annual_Income_inThousandUSD')


# In[23]:


df["Annual_Income_inThousandUSD"] = pd.to_numeric(df["Annual_Income_inThousandUSD"])
df["Spending_Score"] = pd.to_numeric(df["Spending_Score"])


# In[26]:


df.astype({'Annual_Income_inThousandUSD': 'float'}).dtypes
print(df.dtypes)


# In[27]:


df["Annual_Income_inThousandUSD"].dtypes


# In[22]:


a=df.copy()
#z=a.drop(df.columns[1], axis = 1, inplace=True)


# # Drop Once

# In[23]:


#a=a.drop(df.columns[1], axis = 1)
a.head()


# In[24]:


clustering1.fit(a[['Annual_Income_inThousandUSD']])


# In[25]:


clustering1.labels_


# In[26]:


df["Income_cluster"]=clustering1.labels_


# In[27]:


df["Income_cluster"].value_counts()


# In[28]:


clusters = a.copy()
clusters['cluster_pred']=clustering1.fit_predict(a)


# In[29]:


clusters['cluster_pred'].value_counts()


# In[30]:


clusters.head(5)


# In[31]:


df.head()


# In[32]:


plt.figure(figsize=(5,5))
plt.scatter(clusters['Annual_Income_inThousandUSD'],clusters['Spending_Score'],c=clusters['cluster_pred'],cmap='rainbow')
plt.title("Clustering customers based on Annual Income and Spending score", fontsize=15,fontweight="bold")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


# In[94]:


inertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(a)
    inertia_scores.append(kmeans.inertia_)


# In[95]:


plt.plot(range(1,11),inertia_scores)


# In[34]:


df.groupby("Income_cluster")["Age","Annual_Income_inThousandUSD","Spending_Score"].mean()


# In[35]:


score = silhouette_score(a, clustering1.labels_, metric='euclidean')


# In[36]:


print(score)


# In[37]:


score = silhouette_score(clusters, clustering1.labels_, metric='euclidean')
print(score)


# In[38]:


avg_data = df.groupby(['Income_cluster'],
as_index=False).mean()
print(avg_data)


# # Bivariate Clustering

# In[39]:


clustering2=KMeans(5)
clustering2.fit(a[['Annual_Income_inThousandUSD','Spending_Score']])
df["Income_nd_SpendingScore_cluster"]=clustering2.labels_
df.head()


# In[45]:


inertia_score2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(a[['Annual_Income_inThousandUSD','Spending_Score']])
    inertia_score2.append(kmeans2.inertia_)


# In[46]:


plt.plot(range(1,11),inertia_score2)


# In[40]:


df["Income_nd_SpendingScore_cluster"].value_counts()


# In[41]:


centers=pd.DataFrame(clustering2.cluster_centers_)
centers.columns=['x','y']


# In[42]:


plt.figure(figsize=(8,6))
plt.scatter(x=centers['x'],y=centers['y'],s=50,c='black',marker='*')
sns.scatterplot(data=df,x='Annual_Income_inThousandUSD', y='Spending_Score', hue='Income_nd_SpendingScore_cluster',palette='tab10')


# In[43]:


pd.crosstab(df['Income_nd_SpendingScore_cluster'],df['Gender'],normalize='index')


# In[44]:


df.groupby("Income_nd_SpendingScore_cluster")["Age","Annual_Income_inThousandUSD","Spending_Score"].mean()


# In[45]:


score = silhouette_score(a, clustering2.labels_, metric='euclidean')


# In[46]:


print(score)


# # Multivariate Clustering

# In[47]:


from sklearn.preprocessing import StandardScaler


# In[48]:


scale= StandardScaler()


# In[49]:


dff=pd.get_dummies(df,drop_first=True)
dff.head()


# In[50]:


dff.columns


# In[51]:


dff= dff[['Age', 'Annual_Income_inThousandUSD', 'Spending_Score', 'Gender_Male']]
dff.head()


# In[52]:


dff= scale.fit_transform(dff)


# In[53]:


dff=pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[143]:


inertia_score3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    inertia_score3.append(kmeans3.inertia_)
plt.plot(range(1,11),inertia_score2)


# In[54]:


df.to_csv('Clusteringdata.csv')


# In[55]:


clustering3=KMeans(5)
clustering3.fit(dff)
dff["multi_cluster"]=clustering3.labels_
dff.head()


# In[56]:


dff["multi_cluster"].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:




