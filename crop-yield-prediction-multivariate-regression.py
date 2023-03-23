#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Reading the dataset and displaying the first 5 rows

df = pd.read_csv("/kaggle/input/crop-yield-prediction-dataset/yield_df.csv")
df.head()


# In[3]:


df.info()


# In[4]:


# Finding the number if unique values in each column

df.nunique()


# In[5]:


# We will drop the "Unnamed: 0" column

df.drop(columns = ["Unnamed: 0"], inplace = True)
df.head()


# In[6]:


# First let us categorize our features and target variables

features = df.drop(columns = ["hg/ha_yield"])
target = df["hg/ha_yield"]

features.head()


# In[7]:


df["Item"].unique()


# In[8]:


features["Item"] = features["Item"].str.replace(", paddy", "Paddy", regex = False)
features["Item"].unique()


# In[9]:


features["Item"] = features["Item"].str.replace(" and others", "", regex = False)
features["Item"].unique()


# In[10]:


# None of our columns contain low or high cardinality so we do not need to drop any of them

features.select_dtypes("object").head()


# In[11]:


features.select_dtypes("number").head()


# In[12]:


# Let us find features with high-corelation to each other

corr = features.select_dtypes("number").corr()
sns.heatmap(corr)


# In[13]:


# From the above heatmap it is plain to see none of the features have a high co-relation to each other

features.info()


# In[14]:


# We can drop "Area" column as well because when we do one-hot encoding, the 101 unique values will make our features dataframe have 101 extra columns which will lead to high dimensionality

# features.drop(columns = "Area", inplace = True)
features.head()


# In[15]:


# We will be making a pipeline with OneHotEncoder and Ridge Regression

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

model =  make_pipeline(
    OneHotEncoder(),
    Ridge()
)


# In[16]:


model.fit(features, target)


# In[17]:


features.head()


# In[18]:


predictions = model.predict(features)
predictions = pd.Series(predictions)
predictions.head()


# In[19]:


target.head()


# In[20]:


from sklearn.metrics import mean_absolute_error as mae
print("Mean Absolute Error:", mae(target, predictions))


# OneHotEncoder + Ridge = 29110.6 <br>
# OneHotEncoder + LinearRegression = 29144.0

# In[21]:


# What if we don't drop "Area" Column?
# We forgot to build a Baseline Model so let's do that now

avg = target.agg(np.mean)
avg


# In[22]:


baseline_predictions = [avg] * len(target)
len(baseline_predictions)


# In[23]:


print("Mean Absolute Error:", mae(target, baseline_predictions))


# Our Best Model has a MAE of 29110.6 and our Baseline Model has a MAE of 64242.0

# In[24]:


plt.plot(predictions)
plt.plot(target)

plt.show()

