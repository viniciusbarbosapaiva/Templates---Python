# Hierarchical Clustering

#%reset -f
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset
df = pd.read_csv('appdata10.csv')
df_user = pd.DataFrame(np.arange(0,len(df)), columns=['user'])
df = pd.concat([df_user, df], axis=1)
df.info()
df.head()
df.tail()
df.columns.values

#Converting columns to Datatime 
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
time_new = df['Timestamp'].iloc[0]
df['Hour'] = df['Timestamp'].apply(lambda time_new: time_new.hour)
df['Month'] = df['Timestamp'].apply(lambda time_new: time_new.month)
df['Day'] = df['Timestamp'].apply(lambda time_new: time_new.dayofweek)
df["hour"] = df.hour.str.slice(1, 3).astype(int)

#Data analysis
statistical = df.describe()

#Verifying null values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.isna().any()
df.isna().sum()

#Define X
X = df.drop(['enrolled','user','first_open','enrolled_date', 'enrolled_date' ], axis=1)

#Taking care of missing data
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3] )
'''

#Encoding categorical data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_x = LabelEncoder()
X.iloc[:, 1] = labelenconder_x.fit_transform(X.iloc[:, 1])
onehotencoder_x = OneHotEncoder(categorical_features=[1])
X2 = pd.DataFrame(onehotencoder_x.fit_transform(X).toarray())
y = pd.DataFrame(labelenconder_x.fit_transform(y))

#Dummies Trap
X2 = X2.iloc[:, 1:]
X2 = X2.iloc[:,[0,1,2]]
X2 = X2.rename(columns={1:'pay_schedule_1', 2:'pay_schedule_2', 3:'pay_schedule_3'})
X = pd.concat([X,X2], axis=1)
X = X.drop(['pay_schedule'], axis=1)
'''

#Visualizing data
sns.pairplot(data=df, hue='target_names', vars= ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.countplot(x='target_names', data=df, label='Count')
sns.scatterplot(x='mean area', y='mean smoothness',hue='target_names', data=df)
plt.figure(figsize=(20,10))
sns.heatmap(data=df.corr(), annot=True, cmap='viridis')

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

  

