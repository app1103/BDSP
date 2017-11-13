
# coding: utf-8

# # Importing Python Machine Learning Libraries

# In[10]:

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# # Data Import

# In[2]:

balance_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                           sep= ',', header= None)


# In[4]:

print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)


# In[5]:

print ("Dataset:: ")
balance_data.head()


# # Data Slicing

# In[6]:

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]


# In[7]:

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# # Decision Tree Training

# In[8]:

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[9]:

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# # Prediction

# In[11]:

clf_gini.predict([[4, 4, 3, 3]])


# In[12]:

y_pred = clf_gini.predict(X_test)
y_pred


# # Calculating Accuracy Score

# In[14]:

print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)


# In[ ]:



