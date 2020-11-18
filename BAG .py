#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Function importing Dataset
def import_train(dirr):
    # Printing the dataswet shape
    data = pd.read_csv(dirr, sep= ',')
    boot = [resample(data, replace=True, n_samples=2000, random_state = 1) for _ in range(1000)]

    clf_entropy = []
    for i,data in enumerate(boot):
        # Separating the target variable
        X = data.values[:, 0:20]
        Y = data.values[:, -1]

        # Splitting the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size = 0.3)


        # Decision tree with entropy
        entropy = DecisionTreeClassifier(
                criterion = "entropy",
                max_depth = 3, min_samples_leaf = 5)

        # Performing training
        entropy.fit(X_train, y_train)
        clf_entropy.append(entropy)
    return clf_entropy, X_test, y_test


# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    return y_pred

# Driver code
def BAG():
    agg = []
    pred = []
    # Building Phase
    clf_entropy, X_test, y_test = import_train("C:\\Users\\Monty\\Desktop\\BAG\\train.csv")
    # prediction
    for i in range(1000):
        y_pred_entropy = prediction(X_test, clf_entropy[i])
        agg.append(y_pred_entropy)
    agg = np.array(agg)
    agg = agg.transpose()

    for votes in agg:
        votes = list(votes)
        pred.append(max(set(votes), key=votes.count))
    acc = accuracy_score(y_test, pred)*100

    return pred, y_test, acc


# In[3]:


# Function importing Dataset
def import_train_tree(dirr):
    # Printing the dataswet shape
    data = pd.read_csv(dirr, sep= ',')
    # Separating the target variable
    X = data.values[:, 0:20]
    Y = data.values[:, -1]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3)


    # Decision tree with entropy
    entropy = DecisionTreeClassifier(
                criterion = "entropy",
                max_depth = 3, min_samples_leaf = 5)

    # Performing training
    entropy.fit(X_train, y_train)
    return entropy, X_test, y_test


# Function to make predictions
def prediction_tree(X_test, clf_object):

    # Predicton on test with entropy Index
    y_pred = clf_object.predict(X_test)
    return y_pred

# Driver code
def tree():

    # Building Phase
    clf_entropy, X_test, y_test = import_train_tree("C:\\Users\\Monty\\Desktop\\BAG\\train.csv")
    # prediction
    y_pred_entropy = prediction_tree(X_test, clf_entropy)

    acc = accuracy_score(y_test, y_pred_entropy)*100

    return y_pred_entropy, y_test, acc


# In[4]:


# Function importing Dataset
def import_train_rf(dirr):
    # Printing the dataswet shape
    data = pd.read_csv(dirr, sep= ',')
    # Separating the target variable
    X = data.values[:, 0:20]
    Y = data.values[:, -1]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3)


    # Decision tree with entropy
    entropy = RandomForestClassifier(
                criterion = "entropy",
                max_depth = 3, min_samples_leaf = 5)

    # Performing training
    entropy.fit(X_train, y_train)
    return entropy, X_test, y_test


# Function to make predictions
def prediction_rf(X_test, clf_object):

    # Predicton on test with entropy Index
    y_pred = clf_object.predict(X_test)
    return y_pred

# Driver code
def rf():
    # Building Phase
    clf_entropy, X_test, y_test = import_train_rf("C:\\Users\\Monty\\Desktop\\BAG\\train.csv")
    # prediction
    y_pred_entropy = prediction_rf(X_test, clf_entropy)

    acc = accuracy_score(y_test, y_pred_entropy)*100

    return y_pred_entropy, y_test, acc


# In[5]:


acc_bag = np.zeros(100)
acc_tree = np.zeros(100)
acc_rf = np.zeros(100)
for i in range(100):

    pred_b,y_b,acc_b  = BAG()
    acc_bag[i] = acc_b

    pred_t,y_t,acc_t  = tree()
    acc_tree[i] = acc_t

    pred_rf,y_rf,acc_r  = rf()
    acc_rf[i] = acc_r


# In[6]:


x = np.linspace(0,100,100)

plt.plot(x,acc_bag,label = "BAG")
plt.plot(x,acc_tree, label = "Tree")
plt.plot(x,acc_rf, label = "Random Forest")

plt.ylim(0,100)
plt.xlim(0,100)
plt.xlabel("Trial Number")
plt.ylabel("Percent accuracy")
plt.title("Classifier test accuracy over 100 trials.")
plt.legend()
plt.show();

print(f"BAG variance in accuracy: {np.var(acc_bag)}")
print(f"Tree variance in accuracy: {np.var(acc_tree)}")
print(f"Radom Forest variance in accuracy: {np.var(acc_rf)}")
print(" ")

print(f"BAG average accuracy: {np.mean(acc_bag)}")
print(f"Tree average accuracy: {np.mean(acc_tree)}")
print(f"Random Forest average accuracy: {np.mean(acc_rf)}")


# In[ ]:
