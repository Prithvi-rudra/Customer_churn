#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import svm
from sklearn.datasets import load_iris


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import Image
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import seaborn as sns


# In[10]:


df = pd.read_csv('bigml_59c28831336c6604c800002a.csv')

print (df.shape)


# In[11]:


# Load data
df.head(3)


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Count the occurrences of each value in the 'churn' column
y = df["churn"].value_counts()

# Create a bar plot
sns.barplot(x=y.index, y=y.values)

# Optionally, show the plot
plt.xlabel("Churn Status")  # Label for the x-axis
plt.ylabel("Count")          # Label for the y-axis
plt.title("Churn Counts")    # Title for the plot
plt.show()                   # Display the plot


# In[14]:


y_True = df["churn"][df["churn"] == True]
print ("Churn Percentage = "+str( (y_True.shape[0] / df["churn"].shape[0]) * 100 ))


# # Descriptive Analysis

# In[15]:


df.describe()


# # Churn By State

# In[16]:


df.groupby(["state", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10)) 


# # Churn By Area Code

# In[17]:


df.groupby(["area code", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5))


# # Churn By Customers with International plan

# In[18]:


df.groupby(["international plan", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 


# # Churn By Customers with Voice mail plan

# In[19]:


df.groupby(["voice mail plan", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 


# # Handle Categorical Cols - Label Encode

# In[21]:


# Discreet value integer encoder
label_encoder = preprocessing.LabelEncoder()


# In[22]:


# State is string and we want discreet integer values
df['state'] = label_encoder.fit_transform(df['state'])
df['international plan'] = label_encoder.fit_transform(df['international plan'])
df['voice mail plan'] = label_encoder.fit_transform(df['voice mail plan'])

#print (df['Voice mail plan'][:4])
print (df.dtypes)


# In[23]:


df.shape


# In[25]:


df.head()


# # Strip of Response values

# In[31]:


# Convert the 'churn' column to a NumPy array of integers
y = df['churn'].to_numpy().astype(int)

# Get the size of the array
y_size = y.size
print(y_size)


# # Strip off Redundant cols

# In[32]:


# df = df.drop(["Id","Churn"], axis = 1, inplace=True)
df.drop(["phone number","churn"], axis = 1, inplace=True)


# In[34]:


df.head(3)


# # Build Feature Matrix

# In[37]:


# Convert the DataFrame to a NumPy array of floats
X = df.to_numpy().astype(float)

# Check the shape of the array
print(X.shape)


# In[38]:


X


# In[39]:


X.shape


# # Standardize Feature Matrix values

# In[40]:


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


# In[41]:


X


# # Stratified Cross Validation - Since the Response values are not balanced

# In[42]:


def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    # ii -> train
    # jj -> test indices
    for ii, jj in stratified_k_fold: 
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred


# # Build Models and Train

# In[46]:


import numpy as np
from sklearn import ensemble, neighbors, linear_model, metrics, svm
from sklearn.model_selection import StratifiedKFold

def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    y_pred = np.zeros(y.shape)
    
    for train_index, test_index in stratified_k_fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        
        # Initialize the classifier
        clf = clf_class(**kwargs)
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred[test_index] = clf.predict(X_test)
    
    return y_pred

# Example usage with classifiers
print('Gradient Boosting Classifier:  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support Vector Machine (SVM):   {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:      {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:           {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))))


# # Confusion Matrices for various models

# In[50]:


grad_ens_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))
sns.heatmap(grad_ens_conf_matrix, annot=True,  fmt='');
title = 'Gradient Boosting'
plt.title(title);


# In[49]:


svm_svc_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, svm.SVC))
sns.heatmap(svm_svc_conf_matrix, annot=True,  fmt='');
title = 'SVM'
plt.title(title);


# In[51]:


random_forest_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.RandomForestClassifier))
sns.heatmap(random_forest_conf_matrix, annot=True,  fmt='');
title = 'Random Forest'
plt.title(title);


# In[52]:


k_neighbors_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))
sns.heatmap(k_neighbors_conf_matrix, annot=True,  fmt='');
title = 'KNN'
plt.title(title);


# In[54]:


logistic_reg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.LogisticRegression))
sns.heatmap(logistic_reg_conf_matrix, annot=True,  fmt='');
title = 'Logistic Regression'
plt.title(title);


# # classification_report

# In[55]:


print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support vector machine(SVM):\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, linear_model.LogisticRegression))))


# # Final Model Selection
# Gradient Boosting seems to do comparatively for this case

# In[56]:


gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X, y)


# In[57]:


# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
print (gbc.feature_importances_)
feat_importances = pd.Series(gbc.feature_importances_, index=df.columns)
feat_importances = feat_importances.nlargest(19)
feat_importances.plot(kind='barh' , figsize=(10,10)) 


# In[ ]:




