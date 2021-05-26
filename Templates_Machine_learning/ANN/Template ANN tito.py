import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')
sns.set(style='darkgrid', palette='deep')
bins = range(0,100,10)

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

#Converting string format to Datatime format 
import date_converter

for i in range(0,len(df)):
    df['Timestamp'][i] = date_converter.string_to_datetime(df['Timestamp'][i], '%Y-%m-%d %H:%M:%S')
time_new = df['Timestamp'].iloc[0]
df['Hour'] = df['Timestamp'].apply(lambda time_new: time_new.hour)
df['Month'] = df['Timestamp'].apply(lambda time_new: time_new.month)
df['Day'] = df['Timestamp'].apply(lambda time_new: time_new.weekday())

#Visualising Data
def bar_chart(feature1, feature2):
    g = pd.crosstab(df[feature1], df[feature2]).plot(kind='bar', figsize=(10,10), rot = 45)
    ax = g.axes
    for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points') 
    plt.grid(b=True, which='major', linestyle='--')
    plt.legend(['Clicked on Ad',"Did not Clicked on Ad"])
    plt.title('Clicked on Ad for {}'.format(feature1))
    plt.xlabel('{}'.format(feature1))
    plt.tight_layout()
    plt.ylabel('Quantity')
    
def bar_chart_group(feature):
    g = pd.crosstab(pd.cut(df[feature], bins), df['Clicked on Ad']).plot(kind='bar', figsize=(10,10), rot = 45)
    ax = g.axes
    for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points') 
    plt.grid(b=True, which='major', linestyle='--')
    plt.legend(['Clicked on Ad',"Did not Clicked on Ad"])
    plt.title('Clicked on Ad for {}'.format(feature))
    plt.xlabel('{}'.format(feature))
    plt.tight_layout()
    plt.ylabel('Quantity')

def bar_chart_hour(feature):
    bins_hour = np.arange(0,25,12)
    g = pd.crosstab(pd.cut(df[feature], bins_hour), df['Clicked on Ad']).plot(kind='bar', figsize=(10,10), rot = 45)
    ax = g.axes
    for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points') 
    plt.grid(b=True, which='major', linestyle='--')
    plt.legend(['Clicked on Ad',"Did not Clicked on Ad"])
    plt.title('Clicked on Ad for {}'.format(feature))
    plt.xlabel('{}'.format(feature))
    plt.tight_layout()
    plt.ylabel('Quantity')
    
bar_chart('Male','Clicked on Ad')
bar_chart('continent', 'Clicked on Ad')
bar_chart('Day', 'Clicked on Ad')
bar_chart('Month', 'Clicked on Ad')
bar_chart_group('Age')
bar_chart_group('% spending time')
bar_chart_hour('Hour')

df.drop(['user', 'Male', 'Clicked on Ad'], axis=1).hist(figsize=(10,10))
df.groupby('continent')['Area Income'].sum().sort_values().plot(kind='bar', figsize=(10,10), rot=45)
plt.title('Area income per Continent')
plt.grid(b=True, which='major', linestyle='--')
plt.tight_layout()
plt.ylabel('Quantity')

## Correlation with independent Variable 
df2 = df.drop(['user', 'Clicked on Ad', 'Ad Topic Line', 'City'], axis=1)
df2.corrwith(df['Clicked on Ad']).plot.bar(
        figsize = (10, 10), title = "Correlation with Clicked on Ad", fontsize = 15,
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

## Pie Plots 
df.columns
df2 = df.drop(['user', 'Daily Time Spent on Site', 'Daily Internet Usage',
       '% spending time', 'Age', 'Area Income', 'Ad Topic Line', 'City' , 'Country',
       'Timestamp', 'Hour', 'Clicked on Ad'], axis=1)
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

#Data Analysis
statistical = df.describe()    
clicked = df['Clicked on Ad'].value_counts()
countNotClicked = len(df[df['Clicked on Ad'] == 0])     
countClicked  = len(df[df['Clicked on Ad'] == 1]) 
print('Percentage of not Clicked on Ad: {:.2f}%'.format((countNotClicked/len(df)) * 100)) 
print('Percentage of Clicked on Ad: {:.2f}%'.format((countClicked/len(df)) * 100))
df.groupby(df['Clicked on Ad']).mean().head()

#Looking for Null Values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.isnull().any()
df.isnull().sum()
null_percentage = (df.isnull().sum()/len(df) * 100)
null_percentage = pd.DataFrame(null_percentage, columns = ['Percentage Null Values (%)'])
null_percentage

#Define X and y
df.columns
X = df.drop(['user', 'Clicked on Ad', 'Ad Topic Line', 'City',
              'Country', 'Timestamp'], axis=1)
y = df['Clicked on Ad']

#Get Dummies
X = pd.get_dummies(X)

#Avoiding Dummies Trap
X.columns
X = X.drop(['continent_not found'], axis=1)
X.isnull().sum()

#Splitting the Dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0) 
X_train.shape     
X_test.shape     
y_train.shape     
y_test.shape     
y_train.value_counts()
y_test.value_counts()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)
X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)

#Importing Keras libraries e packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import LeakyReLU
leaky_relu_alpha = 0.1
import time
from keras.optimizers import Adam, Adamax, Nadam, SGD
from keras import regularizers

#How many layer and neurons I will use in my model?
'''
def create_model(layers,activation,optimizer):
    classifier = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            classifier.add(Dense(nodes,input_dim = int(X_train.shape[1])))
            classifier.add(Activation(activation))
        else:
            classifier.add(Dense(nodes))
            classifier.add(Activation(activation))
    classifier.add(Dense(1)) #Note: no activation beyond this point.
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn=create_model, batch_size = 10, epochs = 100, verbose=0 )
classifier

layers = [[24,48], [48,24],[22,44],[32,64]]
activation = ['sigmoid', 'relu']
parameters = {'batch_size': [128, 256],
              'layers' : layers,
              'activation' : activation,
              'epochs': [100, 500],
              'optimizer': ['adam', 'sgd']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy, best_parameters    
'''
def create_model(layers,dropout):
    classifier = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            classifier.add(Dense(nodes,init = 'uniform', activation='relu', input_dim = int(X_train.shape[1])))
            classifier.add(Dropout(p= dropout))
        else:
            classifier.add(Dense(nodes, init = 'uniform'))
            classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout ))
            
    classifier.add(Dense(1,init = 'uniform', activation = 'sigmoid')) #Note: no activation beyond this point.
    classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=create_model, batch_size = 10, epochs = 100, verbose=0 )
classifier
int((X_test.shape[1]/2)+1)

layers = [[int((X_test.shape[1]/2)+1),24,48], [int((X_test.shape[1]/2)+1),48,24],[int((X_test.shape[1]/2)+1),24],[int((X_test.shape[1]/2)+1),48]]
dropout = [0.2,0.3,0.4,0.5]
parameters = {
              'layers' : layers,
              'dropout': dropout,
              
              }
#'batch_size': [128, 256],
#'epochs': [100, 500]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy, best_parameters

results2 = pd.DataFrame()
for layer1 in np.arange(len(dropout)):
    for layer2 in np.arange(len(dropout)):
        for layer3 in np.arange(len(dropout)):
#Initialising the ANN
            classifier = Sequential()

#Adding the input layer and the first hidden layer
            classifier.add(Dense(output_dim =int((X_test.shape[1]/2)+1) , init = 'uniform', 
                                 activation = 'relu', input_dim = int(X_test.shape[1])))
#classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout[layer1] ))
#output_dim = média da soma do número de variáveis (no caso 30) + 1
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação linear retificada. A mais utilizada.
#Input_dim = número de variáveis (no caso 30)
#Dropout = evita overfitting. Começa com 0.1. Se continuar com overfitting, tentar de 0.2 a 0.5. Nunca maior que 0.5 senão será underfitting

#Adding the second hidden layer
            classifier.add(Dense(output_dim = 24, init = 'uniform'))#activation = 'relu'
            classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout[layer2]))
#output_dim = média da soma do número de variáveis (no caso 30) + 1
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação linear retificada. A mais utilizada.
#Input_dim = Não será mais necessário pq já foi feito no input layer
#Dropout = evita overfitting. Começa com 0.1. Se continuar com overfitting, tentar de 0.2 a 0.5. Nunca maior que 0.5 senão será underfitting

#Adding the third hidden layer
            classifier.add(Dense(output_dim = 48, init = 'uniform'))#, activation = 'relu'
            classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
            classifier.add(Dropout(p= dropout[layer3]))

#Adding the output layer
            classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#output_dim = Como neste caso queremos 1 ou 0, só teremos 1 output_layer. 
#init = Peso. Gera aleatoriamente. Sempre 'uniform'
#activation = função de ativação sigmoid. A mais utilizada para output layer quando são binárias. 
#Input_dim = Não será mais necessário pq já foi feito no input layer

#Compiling the ANN
            adam=Adam(lr=0.0001)
            classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer = algorítmo para selecionar o melhor peso do ANN. 'Adam' é um dos mais utilizados
#loss =  Algorítmo que minimiza as perdas do gradiente descendete estocástico. Como a saida é binária, utilizou binary_crossentropy. Se houver mais que uma variável categorical_crossentropy
#metrics = Padrão

#Fit classifier to the training test
            history = classifier.fit(X_train, y_train, batch_size = len(X_train), epochs = 100, validation_data=(X_test, y_test))
#batch_size = não tem um valor certo. 
#epochs  = não tem um valor certo

#Predicting the test set result
            y_pred = classifier.predict(X_test)
            y_pred = (y_pred > 0.5) #converte em verdadeiro ou falso

            from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results_final = pd.DataFrame([['ANN Dropout {} - {} - {}'.format(dropout[layer1],dropout[layer2],dropout[layer3]), acc, prec, rec, f1]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
            results2 = results2.append(results_final)
            
            # Plot training & validation accuracy values
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy Dropout {} - {} - {}'.format(dropout[layer1],dropout[layer2],dropout[layer3]))
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss Dropout {} - {} - {}'.format(dropout[layer1],dropout[layer2],dropout[layer3]))
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()   
results2.sort_values(by='Accuracy', ascending=False)   
#Salvando planilha tratada
results2.sort_values(by='Accuracy', ascending=False).to_excel('Resltado Final.xlsx')     

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = int((X_test.shape[1]/2)+1), kernel_initializer = 'uniform', activation = 'relu', 
                         kernel_regularizer=regularizers.l1(0.001),input_dim = int(X_test.shape[1])))
    classifier.add(Dropout(p= 0.4))
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform'))
    classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
    classifier.add(Dropout(p= 0.5))
    classifier.add(Dense(units = 48, kernel_initializer = 'uniform'))
    classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
    classifier.add(Dropout(p= 0.4))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
accuracies.mean()
accuracies.std()
print("ANN Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

import pydotplus
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot
plot_model(classifier, to_file='model.png', show_shapes=True, show_layer_names=True)

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_accuracy, best_parameters

# Predicting Test Set
y_pred = grid_search.predict(X_test)
y_pred = (y_pred > 0.5) #converte em verdadeiro ou falso
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['ANN (GS)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

## EXTRA: Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) 

#Making classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)

#### End of Model ####
# Formatting Final Results
df_raw.columns
user_identifier = df['user']
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted'] = y_pred
final_results = final_results[['user', 'Clicked on Ad', 'predicted']].reset_index(drop=True)



