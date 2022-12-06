#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[7]:


data=pd.read_csv("BreastCancerdata.csv")


# In[8]:


data.replace('?',-99999,inplace=True)


# In[9]:


print(data.axes)


# In[13]:


print(data.shape)


# In[14]:


print(data.describe())


# In[16]:


data.hist(figsize=(15,15),grid=0)
plt.show()


# In[18]:


scatter_matrix(data,figsize=(20,20))
plt.show()


# In[21]:


corlte=data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corlte,cmap='viridis',annot=True,linewidth=0.4)
plt.show()


# In[24]:


#importing all column names from data
columns=data.columns.tolist()

#removing the columns which have less corellaion  with data thsat  is data cleaning 
columns=[cln for cln in columns if cln not in ["Class","ID"]]

#Setting up the output value
otpt="Class"

#Setting up X and Y for getting up the output values
x=data[columns]
y=data[otpt]

#verrifying ny printing the shape of the data
print(x.shape)
print(y.shape)


#Here the data we feed to the model is the shape of 699 rows and 9 columns generally it contains the 699 rows and 11 columns but after processig it got reduced to 699*9 matrix


# In[25]:


# for traing and testing the model we need to split the data for both cases 
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(x,y,test_size=0.2)


# In[48]:


# specifing the parametres for testing option
#seed means folds 
seed=10
scoring='accuracy'
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
#Here we are diving inthe ratio of 80:20 that we use 20% of data is used for testing the model


# In[82]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
models=[]
models.append(('SVM',SVC(gamma='auto')))
models.append(('RFC',RandomForestClassifier(max_depth=5,n_estimators=40)))
results=[]
names=[]
for name,model in models:
    kfold=model_selection.KFold(n_splits=10, random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)


# In[51]:


# Let's visualize in algorithm by plotting 
fig=plt.figure()
fig.suptitle('Algorithm Comparision')
tile=fig.add_subplot(111)
plt.boxplot(results)
tile.set_xticklabels(names)
plt.show()


# In[83]:


from sklearn.model_selection import StratifiedKFold
from yellowbrick.base import ModelVisualizer
from yellowbrick.style import resolve_colors
from yellowbrick.exceptions import YellowbrickValueError
from yellowbrick.model_selection import CVScores
_,ax=plt.subplots()
cv=StartifiedKFold(10)
oz=CVScores(RandomForestClassifier(max_depth=5,n_estimators=40),ax=ax,cv=cv,scorinng='accuracy')
oz.fit(X,y)
oz.poof()


# In[80]:


for name,model in models:
    model.fit(X_train,Y_train)
    predic=model.predict(X_test)
    print(name)
    print(accuracy_score(Y_test,predic))
    print(classification_report(Y_test,predic))
    from sklearn.metrics import confusion_matrix
    predict=model.predict(X_test)
    print("============Confusion Matrix=============")
    print(confusion_matrix(Y_test,predic))
    print('\n')
    from sklearn.metrics import f1_score
    fscore=f1_score(Y_test,predic,average='weighted')
    print("F1 Score: ",fscore)
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import matthews_corrcoef
    coh_score=cohen_kappa_score(Y_test,predic)
    print("Kappa Score: ",coh_score)
    MCC=matthews_corrcoef(Y_test,predic)
    print("MCC Score: ",MCC)
    from sklearn import metrics
    cnfo_matrix=metrics.confusion_matrix(Y_test,predic)
    p=sns.heatmap(pd.DataFrame(cnfo_matrix),annot=True,cmap="YlGnBu",fmt='g')
    plt.title('Confusion Matrix',y=1.1)
    plt.ylabel("Actual ")
    plt.xlabel("Predict")
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




