#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


dataset= load_iris()


# In[4]:


print(dataset.DESCR)


# In[7]:


import seaborn as sns 
df=sns.load_dataset('iris')


# In[8]:


dataset.target


# In[10]:


#independent and dependent features
X=df.iloc[:,:-1]
y=dataset.target


# In[13]:


### train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    X,y,test_size=0.33,random_state=42)


# In[15]:


from sklearn.tree import DecisionTreeClassifier


# In[16]:


classifier=DecisionTreeClassifier(criterion='entropy')


# In[19]:


classifier.fit(x_train,y_train)


# In[21]:


x_train.head()


# In[22]:


from sklearn import tree
plt.figure(figsize=(12,10))
tree.plot_tree(classifier,filled=True)


# In[24]:


## Post Prunning
classifier=DecisionTreeClassifier(criterion='entropy',max_depth=2)
classifier.fit(x_train,y_train)


# In[25]:


from sklearn import tree
plt.figure(figsize=(12,10))
tree.plot_tree(classifier,filled=True)


# In[27]:


##prediction
y_pred=classifier.predict(x_test)


# In[28]:


y_pred


# In[29]:


from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))


# In[30]:


## decision tree prepruning and hyperparameter tuning for huge data


# In[31]:


import warnings
warnings.filterwarnings('ignore')


# In[32]:


parameter={
 'criterion':['gini','entropy','log_loss'],
  'splitter':['best','random'],
  'max_depth':[1,2,3,4,5],
  'max_features':['auto', 'sqrt', 'log2']

}


# In[33]:


from sklearn.model_selection import GridSearchCV


# In[34]:


classifier=DecisionTreeClassifier()
clf=GridSearchCV(classifier,param_grid=parameter,cv=5,scoring='accuracy')


# In[36]:


clf.fit(x_train,y_train)


# In[37]:


clf.best_params_


# In[39]:


y_pred=clf.predict(x_test)


# In[40]:


from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))


# In[ ]:




