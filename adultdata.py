# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:20:17 2019

@author: Arsh Modak
"""
#%%

import pandas as pd
import numpy as np

#%% 

# Creating a data frame 

adult_df = pd.read_csv(r'E:\ARSH\Imarticus\Python\DataSets\adult_data.csv', header = None, delimiter=' *, *',engine='python')
# adult_df.head()

#%% 

# Giving lables

adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

#adult_df.head()
#%%                  

# PREPROCESSING DATA

adult_df=adult_df.replace(['?'],np.nan)
adult_df.isnull().sum() #finding missing values and 0 because the missing values are not NA

#%% 

# Create a copy of dataframe

adult_df_rev = pd.DataFrame.copy(adult_df)
adult_df_rev.describe(include='all')

#%%  

# Replacing the missing values 
 
for value in['workclass','occupation','native_country']:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0],inplace=True)
    
adult_df_rev.isnull().sum()
#adult_df_rev.head()

#%%

adult_df_rev.dtypes

#%%

# To check each categorie occurs how many times
# The below code is for workclass
adult_df_rev.workclass.value_counts()

#%%

colname = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
colname

#%%

# for preprocessing the data 
from sklearn import preprocessing

le = {}

for x in colname:
    le[x] = preprocessing.LabelEncoder()
    
for x in colname:
    adult_df_rev[x] = le[x].fit_transform(adult_df_rev[x])

#%%
    
adult_df_rev.head

#%%
    
adult_df_rev.dtypes


#%%


#  [:, :] [subsetting rows, subsetting cols]. Here ":" is to fetch all rows or cols.
# -1 since income is the last, so in stead of using 15, we use -1
Y = adult_df_rev.values[:, -1] # taking all rows and the last col i.e income
X = adult_df_rev.values[:, : -1] # taking all rows and all cols except income

#%%

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
print(X)

#%%

# np.setprintoptions(threshold = np.inf) # to show all values in the console since by default it shows limited vals.

#%%

# to solve unknown object unknown since we have to convert object to int
Y = Y.astype(int)

#%%

# Splitting the dataset into test and train

from sklearn.model_selection import train_test_split

# random_state is identical to set.seed in R
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10)

#%%

from sklearn.linear_model import LogisticRegression

# Create a model
classifier = LogisticRegression()

# Fitting Training data into the model
classifier.fit(X_train, Y_train) # fit is used to train the data  classifier.fit(dependent, independent)

Y_pred = classifier.predict(X_test)
# print(list(zip(Y_test, Y_pred)))

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: ")
print()
print(cfm)

print()
print()
print("Classification Report:" )
print()
print(classification_report(Y_test, Y_pred))

print()
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy of the Model: ", acc)

#%%

# Adjusting the Threshold for the Confusion Matrix. By Default it is 0.5

# store the predicted probablities

y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

y_pred_class = []
for value in y_pred_prob[:, 1]:  # 1 for class one, i.e the 1st class of the array
    if value > 0.45:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

#%%

"""y_pred_class = []
for value in y_pred_prob[:, 1]:  # 0 for class zero, i.e the 0th class of the array
    if value > 0.6:
        y_pred_class.append(0)
    else:
        y_pred_class.append(1)
print(y_pred_class) """

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, y_pred_class)
print("Confusion Matrix: ")
print()
print(cfm)

print()
print()
print("Classification Report:" )
print()
print(classification_report(Y_test, y_pred_class))

print()
acc = accuracy_score(Y_test, y_pred_class)
print("Accuracy of the Model: ", acc)

#%%

# TO FIND BEST THRESHOLD MANUALLY 

for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)  # if condition is true then 1, if not then 0
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at Threshold ", a, ":", total_err, " , Type II Error: ", 
              cfm[1,0]," , Type I Error: ", cfm[0,1])
    
#%%
    
# ROC CURVE
    
from sklearn import metrics

# For the threshold we gave:
#  fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_class)

# For multiple thresholds:  
fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_prob[:, 1])
auc = metrics.auc(fpr, tpr)

print("Area Under Curve: ", auc)
print()
print("False Positive Rate: ", fpr)
print()
print("True Positive Rate: ", tpr)

# More the AUC value the better the model

# If AUC value is between 0.5 to 0.6, it is a poor model
# If AUC value is between 0.6 to 0.7, it is a bad model
# If AUC value is between 0.7 to 0.8, it is a good model
# If AUC value is between 0.8 to 0.9, it is a very good model
# If AUC value is between 0.9 to 1.0, it is an excellent model


#%%

import matplotlib.pyplot as plt

plt.title("Receiver Operating Characteristics")
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--') # to print the red dotted diagonal line
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

# Better the AUC Value, the more it will be away (upwards) from the diagonal line.
# The more the elbow point is towards the top left corner, the better the value

# This type of metric is used when: you have adjusted your threshold to find which is the best threshold.
# To test this threshold we use AUC. The better the AUC value, more suitable is the threshold.

#%%

# Using CROSS VALIDATION

classifier=(LogisticRegression()) # since were using Logistic Regression

from sklearn import cross_validation

# Performing K-Fold_Cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10) #(no. of observations for training, no. of folds, i.e. K)
print(kfold_cv)

# Running the model using scoring metric as accuracy
# (estimator = which algorithm, i.e. classifier, Independent Variable, Dependent Variable, Model to be used) :
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print("Accuracy for each Iteration: \n", kfold_cv_result)

print()

# Finding the mean to get the overall accuracy. 
print("Mean Accuracy: ", kfold_cv_result.mean())


# If you find your cross validation model is better, we use the below code: 
# We iterate through the K-Fold Model, hence the for loop.
# train_value and test_value are the iterator variables to traverse through the model
# classifier.fit is used for training purpose.
# X_train[train_value] and Y_train[train_value] works on 9 folds
# X_train[test_value] is used on the 10th fold. We use X_train because Cross Validation is done only on Training Data
for train_value, test_value in kfold_cv:
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])

# Below LoC is used for predicting on testin data.
Y_pred1=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))
    

#%%


# Metrics used for the aforementioned Cross Validation Code : 

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred1)
print("Confusion Matrix: ")
print()
print(cfm)

print()
print()
print("Classification Report:" )
print()
print(classification_report(Y_test, Y_pred1))

print()
acc = accuracy_score(Y_test, Y_pred1)
print("Accuracy of the Model: ", acc)

#%%

# RECURSIVE FEATURE ELIIMNATION (RFE)

from sklearn.feature_selection import RFE
rfe = RFE(classifier, 7)

colname1 = adult_df_rev.columns[:]
# Training the Model using fit :
model_rfe = rfe.fit(X_train, Y_train)

print("Num Features: ", model_rfe.n_features_)
print("Selected Features: ")

print(list(zip(colname1, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_)

#%%

Y_pred2 = model_rfe.predict(X_test)
#print(list(zip(Y_test, Y_pred2)))

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred2)
print("Confusion Matrix: ")
print()
print(cfm)

print()
print()
print("Classification Report:" )
print()
print(classification_report(Y_test, Y_pred2))

print()
acc = accuracy_score(Y_test, Y_pred2)
print("Accuracy of the Model: ", acc)


#%%

# We can use the new X and Y which have variables that matter, Scale them, Split the dataset into train and test
# and then perform Logistic Regression on it. 

"""new_data=adult_df_rev[['age','workclass','occupation','race','sex','income']]
new_data.head()
new_X=new_data.values[:,:-1]
new_Y=new_data.values[:,-1]
print(new_X)
print(new_Y)
"""
#%%

# UNIVARIATE FEATURE ELIMINATION USING SELECTKBEST: 

X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k = 11)  # using chisquared test, trial and error by changing value of K
fit1 = test.fit(X, Y)

print(fit1.scores_)
print(list(zip(colname,fit1.get_support())))
X_new = fit1.transform(X)

print(X_new)

#%%

# Scaling the data for UFE (SelectKBest) Since unlike RFE, it does not do it on its own.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_new)

X_new = scaler.transform(X_new)

#%%

# Splitting the data for UFE (SelectKBest) Since unline RFE, it does not do it on its own

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size = 0.3, random_state = 10)

#%%

# Performing Logistic Regression for X_new (contd UFE)

from sklearn.linear_model import LogisticRegression

# Create a model
classifier = LogisticRegression()

# Fitting Training data into the model
classifier.fit(X_train, Y_train) # fit is used to train the data  classifier.fit(dependent, independent)

Y_pred = classifier.predict(X_test)
#print(list(zip(Y_test, Y_pred)))


#%%

# Getting Metrics for above Logistic Regression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: ")
print()
print(cfm)

print()
print()
print("Classification Report:" )
print()
print(classification_report(Y_test, Y_pred))

print()
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy of the Model: ", acc)

#%%

from sklearn.feature_selection import VarianceThreshold

#scaling required
vt = VarianceThreshold(0.3)
fit1 = vt.fit(X, Y)
print(fit1.variances_)

features = fit1.transform(X) # here, features is you "new" X
print(features)
print(features.shape[1])

# After running this, scale (standard scaler) the data, split the data and run logistic regression

#%%










