#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[20]:


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


# In[21]:


clf = DecisionTreeClassifier()


# In[22]:


clf.fit(X_train, y_train)


# In[23]:


y_pred = clf.predict(X_test)


# In[24]:


try:
    y_pred = clf.predict(X_test)
except NotFittedError as e:
    print("Model is not fitted yet.")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


# In[25]:


import joblib


# In[31]:


joblib.dump(clf,'decision_tree_model.pkl')


# In[32]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:





# In[33]:


from sklearn.metrics import accuracy_score
import joblib


# In[34]:


loaded_model = joblib.load('decision_tree_model.pkl')


# In[39]:


new_data = [5.1, 3.5, 1.4, 0.2],[6.3, 2.9, 5.6, 1.8],[4.9, 3.0, 1.4, 0.2]


# In[40]:


new_predictions = loaded_model.predict(new_data)


# In[41]:


true_labels = [0, 2, 0]


# In[42]:


accuracy = accuracy_score(true_labels, new_predictions)
print("Accuracy on new data:", accuracy)


# In[ ]:




