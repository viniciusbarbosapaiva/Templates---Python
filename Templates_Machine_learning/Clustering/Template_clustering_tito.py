#K-means Clustering
#%reset -f
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from pyclustertend import hopkins
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

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
features = ['tipo_de_negociacao','percentual_venda', 'quantas_correcoes',
       'quantos_pontos_avancou', 'quantos_pontos_retornados', 'amplitude']
X = df[features]

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

# Hopkins Test
'''
the null hypothesis (no meaningfull cluster) happens when the hopkins test is 
around 0.5 and the hopkins test tends to 0 when meaningful cluster exists in 
the space. Usually, we can believe in the existence of clusters when the 
hopkins score is bellow 0.25.
Here the value of the hopkins test is quite high but one could think there is
 cluster in our subspace. BUT the hopkins test is highly influenced by outliers,
 let's try once again with normalised data.
'''
hopkins(X, X.shape[0])

# Construção do modelo DBSCAN
dbscan = DBSCAN(eps = 0.2, min_samples = 5, metric = 'euclidean')
y_pred = dbscan.fit_predict(X)

# Construção do modelo mean shift
# bandwidth = Comprimento da Interação entre os exemplos, também conhecido como a largura de banda do algoritmo.
bandwidth = estimate_bandwidth(X, quantile = .1, n_samples = 500)
mean_shift = MeanShift(bandwidth = bandwidth, bin_seeding = True)
mean_shift.fit(X)

#Using the Elbow Method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0) #n_init e max_iter são padrões.
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show() #Quando parar de cair exageradamente no gráfico, este será o número de cluster. Neste caso serão 5 cluesters

#Applying the K-means to Dataset
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(X)
print(90*'_')
print("\nCount of features in each cluster")
print(90*'_')
pd.value_counts(kmeans.labels_, sort=False)

# Silhouette Score
labels = modelo_v1.labels_
silhouette_score(pca, labels, metric = 'euclidean')

# Function that creates a DataFrame with a column for Cluster Number
def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P

# Function that creates Parallel Plots
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates
def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')

P = pd_centers(featuresUsed=features, centers=kmeans.cluster_centers_)
P
parallel_plot(P)
y_kmeans = kmeans.fit_predict(X)

#Visualising the clusters
plt.scatter(np.array(X)[y_kmeans == 0, 0], np.array(X)[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(np.array(X)[y_kmeans == 1, 0], np.array(X)[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(np.array(X)[y_kmeans == 2, 0], np.array(X)[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(np.array(X)[y_kmeans == 3, 0], np.array(X)[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(np.array(X)[y_kmeans == 4, 0], np.array(X)[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')  
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300 , c = 'yellow', label = 'Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (R$)')
plt.ylabel('Spending Score (1 - 100)')
plt.legend()  
plt.show()

