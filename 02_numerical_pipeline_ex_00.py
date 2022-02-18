#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.02
# 
# The goal of this exercise is to fit a similar model as in the previous
# notebook to get familiar with manipulating scikit-learn objects and in
# particular the `.fit/.predict/.score` API.

# Let's load the adult census dataset with only numerical variables

# In[1]:


import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")
data = adult_census.drop(columns="class")
target = adult_census["class"]


# In the previous notebook we used `model = KNeighborsClassifier()`. All
# scikit-learn models can be created without arguments, which means that you
# don't need to understand the details of the model to use it in scikit-learn.
# 
# One of the `KNeighborsClassifier` parameters is `n_neighbors`. It controls
# the number of neighbors we are going to use to make a prediction for a new
# data point.
# 
# What is the default value of the `n_neighbors` parameter? Hint: Look at the
# documentation on the [scikit-learn
# website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# or directly access the description inside your notebook by running the
# following cell. This will open a pager pointing to the documentation.

# In[2]:


from sklearn.neighbors import KNeighborsClassifier

get_ipython().run_line_magic('pinfo', 'KNeighborsClassifier')


# Create a `KNeighborsClassifier` model with `n_neighbors=50`

# In[3]:


# Write your code here.
classifier_model = KNeighborsClassifier(n_neighbors=50)


# Fit this model on the data and target loaded above

# In[4]:


# Write your code here.
classifier_model.fit(data, target)


# Use your model to make predictions on the first 10 data points inside the
# data. Do they match the actual target values?

# In[5]:


# Write your code here.
#print(classifier_model.predict([:10])
target_predicted = classifier_model.predict(data)


# In[6]:


target_predicted[:10]


# In[7]:


target[:10]


# In[8]:


target[:10] == target_predicted[:10]


# In[10]:


print(f"Number of correct prediction: "
      f"{(target[:10] == target_predicted[:10]).sum()} / 10")


# In[11]:


(target == target_predicted).mean()


# Compute the accuracy on the training data.

# In[12]:


# Write your code here.
accuracy = classifier_model.score(data, target)
#model_name = model.__class__.__name__

print(f"The test accuracy is "
      f"{accuracy:.3f}")


# Now load the test data from `"../datasets/adult-census-numeric-test.csv"` and
# compute the accuracy on the test data.

# In[15]:


# Write your code here.
adult_census_test = pd.read_csv("../datasets/adult-census-numeric-test.csv")


# In[20]:


target_name = "class"
target_test = adult_census_test[target_name]
data_test = adult_census_test.drop(columns = [target_name, ])


# In[21]:


print(f"The testing dataset contains {data_test.shape[0]} samples and "
      f"{data_test.shape[1]} features")


# In[25]:


accuracy = classifier_model.score(data_test, target_test)
#model_name = model.__class__.__name__

print(f"The test accuracy is "
      f"{accuracy:.3f}")

