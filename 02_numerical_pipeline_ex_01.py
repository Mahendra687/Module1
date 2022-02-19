#!/usr/bin/env python
# coding: utf-8

# # 📝 Exercise M1.03
# 
# The goal of this exercise is to compare the performance of our classifier in
# the previous notebook (roughly 81% accuracy with `LogisticRegression`) to
# some simple baseline classifiers. The simplest baseline classifier is one
# that always predicts the same class, irrespective of the input data.
# 
# - What would be the score of a model that always predicts `' >50K'`?
# - What would be the score of a model that always predicts `' <=50K'`?
# - Is 81% or 82% accuracy a good score for this problem?
# 
# Use a `DummyClassifier` and do a train-test split to evaluate
# its accuracy on the test set. This
# [link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
# shows a few examples of how to evaluate the generalization performance of these
# baseline models.

# In[1]:


import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")


# We will first split our dataset to have the target separated from the data
# used to train our predictive model.

# In[2]:


target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)


# We start by selecting only the numerical columns as seen in the previous
# notebook.

# In[3]:


numerical_columns = [
    "age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]


# Split the data and target into a train and test set.

# In[6]:


from sklearn.model_selection import train_test_split
# Write your code here.
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size = 0.42)


# In[7]:


print(f"Number of samples in testing: {data_test.shape[0]} => "
      f"{data_test.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")


# In[8]:


print(f"Number of samples in training: {data_train.shape[0]} => "
      f"{data_train.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")


# Use a `DummyClassifier` such that the resulting classifier will always
# predict the class `' >50K'`. What is the accuracy score on the test set?
# Repeat the experiment by always predicting the class `' <=50K'`.
# 
# Hint: you can set the `strategy` parameter of the `DummyClassifier` to
# achieve the desired behavior.

# In[9]:


from sklearn.dummy import DummyClassifier

# Write your code here.
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf.fit(data_train, target_train)
DummyClassifier(random_state=0, strategy='most_frequent')
clf.score(data_test, target_test)


# In[10]:


from sklearn.dummy import DummyClassifier

# Write your code here.
clf = DummyClassifier(strategy='prior', random_state=0)
clf.fit(data_train, target_train)
DummyClassifier(random_state=0, strategy='prior')
clf.score(data_test, target_test)


# In[11]:


from sklearn.dummy import DummyClassifier

# Write your code here.
clf = DummyClassifier(strategy='uniform', random_state=0)
clf.fit(data_train, target_train)
DummyClassifier(random_state=0, strategy='uniform')
clf.score(data_test, target_test)


# In[ ]:


from sklearn.dummy import DummyClassifier

# Write your code here.
clf = DummyClassifier(strategy='stratified', random_state=0)
clf.fit(data_train, target_train)
DummyClassifier(random_state=0, strategy='stratified')
clf.score(data_test, target_test)

