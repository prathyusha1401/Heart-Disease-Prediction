# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:10:23 2021

@author: PRATHUSHA
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn.metrics import confusion_matrix

# loading the csv data to a Pandas DataFrame
heart= pd.read_csv('heart.csv')

# checking the distribution of Target Variable
heart['target'].value_counts()

X = heart.drop(columns='target', axis=1)
Y = heart['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=4)

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

#Confusion matrix
y_pred=model.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sn.heatmap(conf_matrix,annot=True,fmt='d',cmap="YlGnBu")
plt.show()

#graph
plt.hist([heart[heart.target==0].age, heart[heart.target==1].age], bins = 20, alpha = 0.5, label = ["no_heart_disease","with heart disease"])
plt.xlabel("age")
plt.ylabel("percentage")
plt.legend()
plt.show()




