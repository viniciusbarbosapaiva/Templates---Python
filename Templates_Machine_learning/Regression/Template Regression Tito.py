#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

#Analysing dataset with padas profiling
#from pandas_profiling import ProfileReport
#profile = ProfileReport(df, title='Medical Cost Personal Datasets', html={'style':{'full_width':True}})

#Importing the dataset
df = pd.read_csv('appdata10.csv')
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

#Define X and y
X = df.drop(['enrolled','user','first_open','enrolled_date', 'enrolled_date' ], axis=1)
y = df['enrolled']

#Get Dummies
X = pd.get_dummies(X)

#Dummies Trap
X.columns
X = X.drop(['Gender_Male', 'Geography_Germany'], axis= 1)

#Taking care of missing data
'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3] )
'''

#Encoding categorical data
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_x = LabelEncoder()
labelenconder_y = LabelEncoder()
X.iloc[:, 1] = labelenconder_x.fit_transform(X.iloc[:, 1])
onehotencoder_x = OneHotEncoder(categorical_features=[1])
X2 = pd.DataFrame(onehotencoder_x.fit_transform(X).toarray())
y = pd.DataFrame(labelenconder_y.fit_transform(y))

#Dummies Trap
X2 = X2.iloc[:, 1:]
X2 = X2.iloc[:,[0,1,2]]
X2 = X2.rename(columns={1:'pay_schedule_1', 2:'pay_schedule_2', 3:'pay_schedule_3'})
X = pd.concat([X,X2], axis=1)
X = X.drop(['pay_schedule'], axis=1)
'''

#Visualizing data
sns.pairplot(data=df, vars= ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.scatterplot(x='mean area', y='mean smoothness',hue='target_names', data=df)
plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(), annot=True, cmap='viridis')
df.plot(kind = 'scatter', x= 'longitude', y='latitude', alpha=0.4,
        s=df['population']/100, label = 'population', figsize=(10,7),
        c= 'median_house_value',cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
df.hist(bins= 50, figsize=(10,10))
plt.show()

## Histograms
df2 = df.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(10, 10))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(df2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(df2.columns.values[i])

    vals = np.size(df2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(df2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Correlation with independent Variable (Note: Models like RF are not linear like these)

df2.corrwith(df.e_signed).plot.bar(
        figsize = (10, 10), title = "Correlation with E Signed", fontsize = 15,
        rot = 45, grid = True)

## Correlation Matrix
sns.set(style="white")

# Compute the correlation matrix
corr = df2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

## Pie Plots (Just for binary values)
df.columns
df2 = df[['home_owner', 'has_debt', 'risk_score_2',
                    'risk_score_3', 'risk_score_4', 'risk_score_5',
                    'ext_quality_score_2', 'ext_quality_score', 'e_signed',
                    'new_month_employed', 'new_personal_account_m',
                    ]]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, df2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(df2.columns.values[i - 1])
   
    values = df2.iloc[:, i - 1].value_counts(normalize = True).values
    index = df2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#Splitting the Dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Balancing the Training Set
y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]

#Feature scaling
from sklearn.preprocessing import StandardScaler
y_test_svr = pd.DataFrame(y_test)
y_train_svr = pd.DataFrame(y_train)
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)
X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
y_test_svr = pd.DataFrame(sc_y.fit_transform(y_test_svr), columns=y_test_svr.columns.values)
y_train_svr = pd.DataFrame(sc_y.transform(y_train_svr), columns=y_train_svr.columns.values)
y_test_svr = y_test_svr.iloc[:,0]
y_train_svr =  y_train_svr.iloc[:,0]

#### Model Building ####
### Comparing Models
## Multiple Linear Regression Regression
from sklearn.linear_model import LinearRegression
k = X_test.shape[1]
n = len(X_test)
lr_regressor = LinearRegression(fit_intercept=True)
lr_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = lr_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

results = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

## Ridge Regression
from sklearn.linear_model import Ridge
rd_regressor = Ridge(alpha=50)
rd_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = rd_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Ridge Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Lasso Regression
from sklearn.linear_model import Lasso
la_regressor = Lasso(alpha=500)
la_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = la_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Lasso Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Polynomial Regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lr_poly_regressor = LinearRegression(fit_intercept=True)
lr_poly_regressor.fit(X_poly, y_train)

# Predicting Test Set
y_pred = lr_poly_regressor.predict(poly_reg.fit_transform(X_test))
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test))*100

model_results = pd.DataFrame([['Polynomial Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Suport Vector Regression 
'Necessary Standard Scaler '
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = svr_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Support Vector RBF', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

'''
# Predicting Test Set
y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(X_test)))
from sklearn import metrics
mae = metrics.mean_absolute_error(sc_y.inverse_transform(y_test), y_pred)
mse = metrics.mean_squared_error(sc_y.inverse_transform(y_test), y_pred)
rmse = np.sqrt(metrics.mean_squared_error(sc_y.inverse_transform(y_test), y_pred))
r2 = metrics.r2_score(sc_y.inverse_transform(y_test), y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Support Vector rbf', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)
'''

## Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = dt_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Decision Tree Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(dt_regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf('titanic.pdf')
graph.write_png('titanic.png')

## Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(X_train,y_train)

# Predicting Test Set
y_pred = rf_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Random Forest Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Ada Boosting
from sklearn.ensemble import AdaBoostRegressor
ad_regressor = AdaBoostRegressor()
ad_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = ad_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['AdaBoost Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = gb_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['GradientBoosting Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Xg Boosting
from xgboost import XGBRegressor
xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = xgb_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Ensemble Voting regressor
from sklearn.ensemble import VotingRegressor
voting_regressor = VotingRegressor(estimators= [('lr', lr_regressor),
                                                ('rd', rd_regressor),
                                                ('la', la_regressor),
                                                ('lr_poly', lr_poly_regressor),
                                                ('svr', svr_regressor),
                                                ('dt', dt_regressor),
                                                ('rf', rf_regressor),
                                                ('ad', ad_regressor),
                                                ('gr', gb_regressor),
                                                ('xg', xgb_regressor)])

for clf in (lr_regressor,lr_poly_regressor,svr_regressor,dt_regressor,
            rf_regressor, ad_regressor,gb_regressor, xgb_regressor, voting_regressor):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, metrics.r2_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Ensemble Voting', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)  

#The Best Classifier
print('The best regressor is:')
print('{}'.format(results.sort_values(by='Adj. R2 Score',ascending=False).head(5)))

#Applying K-fold validation
from sklearn.model_selection import cross_val_score
def display_scores (scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard:', scores.std())

lin_scores = cross_val_score(estimator=regressor, X=X_train, y=y_train, 
                             scoring= 'neg_mean_squared_error',cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),
           pd.DataFrame(np.transpose(regressor.coef_), columns = ["coef"])
           ],axis = 1)
    
# Applying Grid Search

# Round 1
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = regressor, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "neg_mean_squared_error",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
# Predicting Test Set
y_pred = grid_search.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Entropy)', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])

results = results.append(model_results, ignore_index = True)

## Feature Selection
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Model to Test
regressor = LinearRegression()

# Select Best X Features
rfe = RFE(regressor, n_features_to_select=None)
rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
X_train.columns[rfe.support_]

# New Correlation Matrix
sns.set(style="white")

# Compute the correlation matrix
corr = X_train[X_train.columns[rfe.support_]].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})    


# Fitting Model to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = regressor.predict(X_test[X_train.columns[rfe.support_]])
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_pred) / y_test  )  )*100

model_results = pd.DataFrame([['Multiple Linear Regression (RF)', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

#Applying K-fold validation
from sklearn.model_selection import cross_val_score
def display_scores (scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard:', scores.std())

lin_scores = cross_val_score(estimator=regressor, X=X_train[X_train.columns[rfe.support_]], 
                             y=y_train, 
                             scoring= 'neg_mean_squared_error',cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),
           pd.DataFrame(np.transpose(regressor.coef_), columns = ["coef"])
           ],axis = 1)   

