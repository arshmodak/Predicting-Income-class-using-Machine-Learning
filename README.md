# Predicting-Income-class-using-Machine-Learning
Predicting Income (<=50 or >50) using Machine Learning on Adult Data:
===========================================================================  
The adult data set consists of various features such as age, sex, education, occupation, relationship status, race etc. 
Using these features, I have predicted if an individuals  income is either &lt;=50k or >50. 
Utilizing libraries such as numpy, pandas, sklearn etc, I created various machine learning  models to predict income. 
I also took care of imputing missing values, feature elimination, label encoding and standardization of data for the process.

Metrics such as Precision, Recall,  F1-Score, Accuracy, True Positive Rate, False Positive Rate and ROC were used to evaluate the model. 

Primarily I used Logistic Regression to predict the income, then I utilized various techniques to improve my model. 
They are as follows:  
1. I manually selected a 0/1 classifier threshold by checking type 1 and type 2 errors.
2. Implemented k-fold cross validation to reduce bias in the models. 
3. Implemented Recursive Feature Elimination 
4. Uni-variate Feature Elimination using "Select K Best"
