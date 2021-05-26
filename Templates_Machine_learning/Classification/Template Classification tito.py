import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import warnings
warnings.filterwarnings('ignore')
sns.set(style='darkgrid', palette='deep')

#Analysing dataset with padas profiling
#from pandas_profiling import ProfileReport
#profile = ProfileReport(df, title='Medical Cost Personal Datasets', html={'style':{'full_width':True}})

#Importing Dataset
df_raw = pd.read_excel('titanic3.xls')
new_columns = ['class','survival', 'name', 'sex', 'age', 'siblings/spouses',
               'parents/children', 'ticket', 'fare', 'cabin', 'embarked', 'lifeboat',
               'body number', 'home/destination']
df_raw.info()

#Feature Engineering
df = pd.DataFrame(df_raw.values, columns= new_columns )
df_user = pd.DataFrame(np.arange(0, len(df)), columns=['passanger'])
df = pd.concat([df_user, df], axis=1)
df['family'] = df['siblings/spouses'] + df['parents/children'] + 1
df = df.drop(['siblings/spouses','parents/children'], axis=1)
df['embarked'].value_counts()
df['embarked'].replace(['S', 'C', 'Q'], 
  ['southampton', 'cherbourg', 'quennstone'], inplace= True )
df.info()
df.columns
df[['class', 'survival', 'age', 'fare',
    'body number', 'family']] = df[['class',  'survival', 'age', 'fare',
    'body number', 'family']].apply(pd.to_numeric)

#Converting columns to Datatime 
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
time_new = df['Timestamp'].iloc[0]
df['Hour'] = df['Timestamp'].apply(lambda time_new: time_new.hour)
df['Month'] = df['Timestamp'].apply(lambda time_new: time_new.month)
df['Day'] = df['Timestamp'].apply(lambda time_new: time_new.dayofweek)
df["hour"] = df.hour.str.slice(1, 3).astype(int)

#Visualising Dataset
bins = range(0,100,10)

ax = sns.distplot(df.age[df.y=='yes'],
              color='red', kde=False, bins=bins, label='Have Subscribed')
sns.distplot(df.age[df.y=='no'],
         ax=ax,  # Overplots on first plot
         color='blue', kde=False, bins=bins, label="Haven't Subscribed")
plt.legend()
plt.show()

g = pd.crosstab(df.sex, df.survival).plot(kind='bar', figsize=(10,5))
ax = g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')  
plt.grid(b=True, which='major', linestyle='--')
plt.title('Survival Frequency for Genre')
plt.legend(['Not Survived', 'Survived'])
plt.xlabel('Genre')
plt.ylabel('Quantity')
plt.show()

df.groupby(pd.cut(df.age, bins))['age'].count().plot(kind='bar', figsize=(10,10))
plt.grid(b=True, which='major', linestyle='--')
plt.title('Frequency of Age')
plt.grid(b=True, which='major', linestyle='--')
plt.xlabel('Age')
plt.ylabel('Quantity')
plt.show()

pd.crosstab(pd.cut(df.age, bins), df.survival).plot(kind='bar', figsize=(10,10))
plt.grid(b=True, which='major', linestyle='--')
plt.title('Survival Frequency for Age')
plt.legend(['Not Survival', 'Survival'])
plt.yticks(np.arange(0,250,50))
plt.xlabel('Age')
plt.ylabel('Quantity')
plt.show()

age_notsurvival = (df.groupby(pd.cut(df.age, bins))['age'].count()/ len(df[df.survival==0]))*100
age_survival = (df.groupby(pd.cut(df.age, bins))['age'].count()/ len(df[df.survival==1]))*100
age_notsurvival.plot(kind='bar', figsize=(10,10))
plt.grid(b=True, which='major', linestyle='--')
plt.title('Percentage of Age for Passanger Not Survived')
plt.yticks(np.arange(0,110,10))
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.show()

age_survival.plot(kind='bar', figsize=(10,10))
plt.grid(b=True, which='major', linestyle='--')
plt.title('Percentage of Age for Passanger Survived')
plt.yticks(np.arange(0,110,10))
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(10,10))
plt.subplots_adjust(hspace=0)
plt.suptitle('Age Frequency')
ax1 = sns.countplot(pd.cut(df.age, bins), data= df, 
                    color='darkblue', ax=axes[0], saturation=0.5)
ax2 = sns.countplot(pd.cut(df.age, bins)[df.survival==0], data=df , 
                    color='red', ax=axes[1], saturation=1, alpha=0.5)
ax2.set_xlabel('Age')
ax3 = sns.countplot(pd.cut(df.age, bins)[df.survival==1], data= df, 
                    color='darkblue', ax=ax2, saturation=1, alpha=0.5)
ax2.legend(['Have Not Survived', 'Have Survived'])

pd.crosstab(df['class'], df.survival).plot(kind='bar', figsize=(15,10))
plt.grid(b=True, which= 'major', linestyle='--')
plt.title('Survival Frequency for Class')
plt.yticks(np.arange(0,600,50))
plt.legend(['Not Survival', 'Survival'])
plt.xlabel('class')
plt.ylabel('Quantity')
plt.show()

pd.crosstab(df.embarked, df.survival).plot(kind='bar', figsize=(15,10))
plt.grid(b=True, which='major', linestyle='--')
plt.yticks(np.arange(0,700,50))
plt.title('Survival Frequency for Embarked')
plt.legend(['Not Survival', 'Survival'])
plt.xlabel('Embarked')
plt.ylabel('Quantity')
plt.show()

sns.pairplot(data=df, hue='survival', vars=['age', 'fare', ])
sns.countplot(x='survival', data=df)
sns.heatmap(data= df.corr(),annot=True,cmap='viridis')
sns.distplot(df.age, bins=10)

pd.crosstab(df.survival[df.embarked=='southampton'],df['class']).plot(kind='bar', figsize=(15,10))
plt.title('Survival Frequency for Class / Embarked(Southampton)')
plt.grid(b=True, which='Major', linestyle='--')
plt.legend(['First Class', 'Second Class', 'Third Class'])
plt.ylabel('Quatity')
plt.xlabel('Survival')
plt.show()

df.drop(['passanger', 'survival'], axis=1).hist(figsize=(10,10))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Correlation with independent Variable (Note: Models like RF are not linear like these)
df2 = df.drop(['passanger', 'name', 'home/destination', 'survival'], axis=1)
df2.corrwith(df.survival).plot.bar(
        figsize = (10, 10), title = "Correlation with Survival", fontsize = 15,
        rot = 45, grid = True)

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
df2 = df[['class','survival','sex', 'embarked']]
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

#Data analysis
statistical = df.describe()    
survival = df.survival.value_counts()
countNotsurvival = len(df[df.survival == 0])     
countSurvival = len(df[df.survival == 1]) 
print('Percentage of Titanic not survival: {:.2f}%'.format((countNotsurvival/len(df)) * 100)) 
print('Percentage of Titanic survival: {:.2f}%'.format((countSurvival/len(df)) * 100))
df.groupby(df['survival']).mean()

#Looking for Null Values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.isnull().any()
df.isnull().sum()
null_percentage = (df.isnull().sum()/len(df) * 100)
null_percentage = pd.DataFrame(null_percentage, columns = ['Percentage Null Values (%)'])
null_percentage

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
#Splitting the Dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
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

X_train = X_train.loc[new_indexes]
y_train = y_train.loc[new_indexes]

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)
X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)

#Applying PCA (If Necessary)
'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)#Antes de sabermos quantas variáveis serão reduzidas, utiliza-se o None primeiro. Depois 
será substituido pelo a quantidade de variáveis com maior variância gerada pela explained_variance. Neste caso, 02.
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_ #Verifica as variáveis com maior variância
'''

#### Model Building ####
### Comparing Models

## Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2')
lr_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = lr_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

## K-Nearest Neighbors (K-NN)
#Choosing the K value
error_rate= []
for i in range(1,40):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print(np.mean(error_rate))

from sklearn.neighbors import KNeighborsClassifier
kn_classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p= 2)
kn_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = kn_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

## SVM (Linear)
from sklearn.svm import SVC
svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)
svm_linear_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = svm_linear_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

## SVM (rbf)
from sklearn.svm import SVC
svm_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)
svm_rbf_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = svm_rbf_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
gb_classifier = GaussianNB()
gb_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = gb_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(X_train, y_train)

#Predicting the best set result
y_pred = dt_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(dt_classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_pdf('titanic.pdf')
graph.write_png('titanic.png')

## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'gini')
rf_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = rf_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

## Ada Boosting
from sklearn.ensemble import AdaBoostClassifier
ad_classifier = AdaBoostClassifier()
ad_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = ad_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Ada Boosting', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

##Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gr_classifier = GradientBoostingClassifier()
gr_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = gr_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Gradient Boosting', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

##Xg Boosting
from xgboost import XGBClassifier
xg_classifier = XGBClassifier()
xg_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = xg_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Xg Boosting', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

##Ensemble Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
voting_classifier = VotingClassifier(estimators= [('lr', lr_classifier),
                                                  ('kn', kn_classifier),
                                                  ('svc_linear', svm_linear_classifier),
                                                  ('svc_rbf', svm_rbf_classifier),
                                                  ('gb', gb_classifier),
                                                  ('dt', dt_classifier),
                                                  ('rf', rf_classifier),
                                                  ('ad', ad_classifier),
                                                  ('gr', gr_classifier),
                                                  ('xg', xg_classifier),],
voting='soft')

for clf in (lr_classifier,kn_classifier,svm_linear_classifier,svm_rbf_classifier,
            gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, xg_classifier,
            voting_classifier):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Ensemble Voting', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)    

#The Best Classifier
print('The best classifier is:')
print('{}'.format(results.sort_values(by='Accuracy',ascending=False).head(5)))

#Applying K-fold validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=gr_classifier, X=X_train, y=y_train,cv=10)
accuracies.mean()
accuracies.std()
print("Gradient Boosting Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

#Plotting Cumulative Accuracy Profile (CAP)
y_pred_proba = classifier.predict_proba(X=X_test)
import matplotlib.pyplot as plt
from scipy import integrate
def capcurve(y_values, y_preds_proba):
    num_pos_obs = np.sum(y_values)
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
    xx = np.arange(num_count) / float(num_count - 1)
    
    y_cap = np.c_[y_values,y_preds_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(level = y_cap_df_s.index.names, drop=True)
    
    print(y_cap_df_s.head(20))
    
    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0
    
    percent = 0.5
    row_index = int(np.trunc(num_count * percent))
    
    val_y1 = yy[row_index]
    val_y2 = yy[row_index+1]
    if val_y1 == val_y2:
        val = val_y1*1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index+1]
        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)
    
    sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
    sigma_model = integrate.simps(yy,xx)
    sigma_random = integrate.simps(xx,xx)
    
    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx,yy, color='red', label='User Model')
    ax.plot(xx,xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')
    
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.25)
    plt.title("CAP Curve - a_r value ="+str(ar_value))
    plt.xlabel('% of the data')
    plt.ylabel('% of positive obs')
    plt.legend()
    plt.savefig('C:\\Users\paiva.SURCO\Documents\Machine Learning\Machine Learning A-Z\Templates\Classification\cap_curve-master\cap_graph.pdf')

capcurve(y_test,y_pred_proba[:,1])

#Plotting ROC curve
from sklearn.metrics import plot_roc_curve
svc_disp = plot_roc_curve(rf_classifier, X_test, y_test)

#Permutation Importance
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(classifier, random_state=0).fit(X_test,y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)

### Parameter Tuning
# Applying Grid Search

# Round 1: Entropy
parameters = {"max_depth": [3, None],
              "max_features": [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters

# Round 2: Entropy
parameters = {"max_depth": [None],
              "max_features": [3, 5, 7],
              'min_samples_split': [8, 10, 12],
              'min_samples_leaf': [1, 2, 3],
              "bootstrap": [True],
              "criterion": ["entropy"]}

from sklearn.metrics import make_scorer
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = scoring,
                           cv = 10,
                           refit = 'AUC',
                           return_train_score=True,
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
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Entropy)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

## EXTRA: Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))   


#### Feature Selection ####


## Feature Selection
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Model to Test
classifier = LogisticRegression(random_state=0)

# Select Best X Features
rfe = RFE(classifier, n_features_to_select=None)
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
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

# Evaluating Results
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(data=cm, annot=True)

#Making the classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train[X_train.columns[rfe.support_]],
                             y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)    
 
#### End of Model ####

# Formatting Final Results
user_identifier = df['user']
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted'] = y_pred
final_results = final_results[['user', 'Clicked on Ad', 'predicted']].reset_index(drop=True)

# Visualising the Training set results (Only two variables)
'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Formatting Final Results
test_identity = X_test['user']
final_results = pd.concat([y_test, test_identity], axis = 1).dropna()
final_results['predicted_reach'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_reach']].reset_index(drop=True)
'''











