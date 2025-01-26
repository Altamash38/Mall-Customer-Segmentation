#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


df= pd.read_csv("D:\Data sets\Mall_Customers.csv")


# In[5]:


df.head()

df.rename(columns = {'Annual Income (k$)':'Annual_Income_inThousandUSD','Spending Score (1-100)':'Spending_Score' }, inplace = True)

df.head()

# Statistical Analysis

df.describe()

sns.distplot(df['Annual_Income_inThousandUSD'])

sns.histplot(df['Annual_Income_inThousandUSD'])

columns=['Age', 'Annual_Income_inThousandUSD','Spending_Score']

for i in columns:
 plt.figure()
 sns.distplot(df[i])

plt.figure(figsize=(8, 6))  # Increase the width and height of the plot
df.boxplot(column=['Age', 'Annual_Income_inThousandUSD', 'Spending_Score'], grid=False)
plt.title("Box Plot of Age, Annual Income and Spend Score")
plt.ylabel("Value")
# plt.xticks(rotation=45)  # If labels are overlapping
plt.show()


from scipy.stats import skew

spend_skewness = skew(df['Spending_Score'])
print("Skewness of Spend Score:", spend_skewness)

income_skewness = skew(df['Annual_Income_inThousandUSD'])
print("Skewness of Annual Income:", income_skewness)

Age_skewness = skew(df['Age'])
print("Skewness of Age:", Age_skewness)

age_bins = [18, 25, 35, 45, 55, 65, 70]  # Define age ranges
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-70']  # Labels for the groups

df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

age_group_counts = df.groupby('Age_Group')['CustomerID'].count()
print(age_group_counts)

income_bins = [15, 46 , 77 , 108 ,137]  # Define age ranges
income_labels = ['15-45', '46-76', '77-107', '108-137']  # Labels for the groups

df['Income_Group'] = pd.cut(df['Annual_Income_inThousandUSD'], bins=income_bins, labels=income_labels, right=False)

income_group_counts = df.groupby('Income_Group')['CustomerID'].count()
print(income_group_counts)

# Exploratory Analysis of Age, Spending, and Income by Gender


# sns.kdeplot(df['Annual_Income_inThousandUSD'],shade=True, hue=df['Gender']);
sns.kdeplot(data=df, x='Annual_Income_inThousandUSD', hue='Gender', shade=True);

df['Gender'].value_counts(normalize=True)

# BI Variate Analysis

for i in columns:
    plt.figure()
    sns.kdeplot(data=df, x=i, hue='Gender', shade=True); # Specify data and x explicitly

#sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Age');
sns.scatterplot(data=df,x='Annual_Income_inThousandUSD',y='Spending_Score');

x=df.copy()
y=x.drop('CustomerID',axis=1)
sns.pairplot(y,hue='Gender');

# df.groupby(['Gender'])['Age', 'Annual_Income_inThousandUSD','Spending_Score'].mean()
df.groupby(['Gender'])[['Age', 'Annual_Income_inThousandUSD','Spending_Score']].mean() # Changed from tuple to list for column selection

df.groupby(['Gender'])[['Age']].describe()

df.groupby(['Gender'])[['Annual_Income_inThousandUSD']].describe()

df.groupby(['Gender'])[['Spending_Score']].describe()

z=y.copy()
z = y.drop(['Gender','Age_Group', 'Income_Group'], axis=1)
sns.heatmap(z.corr(),annot=True);

#Univariate Clustering

dff=df.copy().drop(['Age_Group','Income_Group'],axis=1)
dff.head()

clustering1= KMeans(5)

a=dff.copy().drop(dff.columns[1], axis = 1)
a.head()

clustering1.fit(a[['Annual_Income_inThousandUSD']])

clustering1.labels_

clusters = a.copy()
clusters['cluster_pred']=clustering1.fit_predict(a)

clusters['cluster_pred'].value_counts()

clusters.head(5)

dff["Income_cluster"]=clustering1.labels_

dff.head()

plt.figure(figsize=(5,5))
plt.scatter(clusters['Annual_Income_inThousandUSD'],clusters['Spending_Score'],c=clusters['cluster_pred'],cmap='rainbow')
plt.title("Clustering customers based on Annual Income and Spending score", fontsize=15,fontweight="bold")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

Inertia is simply a measure that suggests how well a data set is clustered

We will find the K numbers of clusters manually by the Elbow method.

The elbow method illustrates that the optimal number of clusters is where the inertia value starts decreasing slowly.

inertia_scores=[]
for i in range(1,11):
 kmeans=KMeans(n_clusters=i)
 kmeans.fit(a)
 inertia_scores.append(kmeans.inertia_)

plt.plot(range(1,11),inertia_scores)

# dff.groupby("Income_cluster")["Age","Annual_Income_inThousandUSD","Spending_Score"].mean()
dff.groupby("Income_cluster")[["Age","Annual_Income_inThousandUSD","Spending_Score"]].mean()

score = silhouette_score(a, clustering1.labels_, metric='euclidean')
print(score)

score = silhouette_score(clusters, clustering1.labels_, metric='euclidean')
print(score)

avg_data = dff.groupby(['Income_cluster'], as_index=False).agg({
    'Age': 'mean',
    'Annual_Income_inThousandUSD': 'mean',
    'Spending_Score': 'mean'
})
print(avg_data)

#Bi Variate Clustering


clustering2=KMeans(5)
clustering2.fit(a[['Annual_Income_inThousandUSD','Spending_Score']])
clusters["cluster_pred2"]=clustering2.labels_
clusters.head()

dff["Income_nd_SpendScore_cluster"]=clustering2.labels_

Finding Inertia Score

inertia_score2=[]
for i in range(1,11):
 kmeans2=KMeans(n_clusters=i)
 kmeans2.fit(a[['Annual_Income_inThousandUSD','Spending_Score']])
 inertia_score2.append(kmeans2.inertia_)


plt.plot(range(1,11),inertia_score2)

clusters["cluster_pred2"].value_counts()

centers=pd.DataFrame(clustering2.cluster_centers_)
centers.columns=['x','y']

plt.figure(figsize=(8,6))
plt.scatter(x=centers['x'],y=centers['y'],s=50,c='black',marker='*')
sns.scatterplot(data=clusters,x='Annual_Income_inThousandUSD', y='Spending_Score', hue='cluster_pred2',palette='tab10')

pd.crosstab(clusters['cluster_pred2'],df['Gender'],normalize='index')

clusters.groupby("cluster_pred2")[["Age","Annual_Income_inThousandUSD","Spending_Score"]].mean()

score = silhouette_score(a, clustering2.labels_, metric='euclidean')
print(score)

# Multivariate Clustering

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

dff['Gender_Male'] = dff['Gender'].replace({'Male': 1, 'Female': 0})

dff.head()

Dum_dff=pd.get_dummies(dff,drop_first=True)
Dum_dff.head()

Dum_dff.columns

Dum_dff= Dum_dff[['Age', 'Annual_Income_inThousandUSD', 'Spending_Score', 'Gender_Male']]
Dum_dff.head()

Dum_dff= scale.fit_transform(Dum_dff)

Dum_dff=pd.DataFrame(scale.fit_transform(Dum_dff))
Dum_dff.head()

inertia_score3=[]
for i in range(1,11):
 kmeans3=KMeans(n_clusters=i)
 kmeans3.fit(Dum_dff)
 inertia_score3.append(kmeans3.inertia_)

plt.plot(range(1,11),inertia_score2)

clustering3=KMeans(5)
clustering3.fit(Dum_dff)

Dum_dff = pd.DataFrame(Dum_dff)
Dum_dff["multi_cluster"]=clustering3.labels_
Dum_dff.head()

Dum_dff["multi_cluster"].value_counts()
