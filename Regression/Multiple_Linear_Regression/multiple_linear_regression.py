#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# <a name="4"></a>
# ## Concept
# 
# Multiple linear regression is used to estimate the relationship between two or more independent variables and one dependent variable. You can use multiple linear regression when you want to know:
# 
# - How strong the relationship is between two or more independent variables and one dependent variable (e.g. how rainfall, temperature, and amount of fertilizer added affect crop growth).
# - The value of the dependent variable at a certain value of the independent variables (e.g. the expected yield of a crop at certain levels of rainfall, temperature, and fertilizer addition).
# 
# ### Assumptions of multiple linear regression
# 
# Multiple linear regression makes all of the same assumptions as simple linear regression:
# 
# - Homogeneity of variance (homoscedasticity): the size of the error in our prediction doesn’t change significantly across the values of the independent variable.
# 
# - Independence of observations: the observations in the dataset were collected using statistically valid sampling methods, and there are no hidden relationships among variables.
#   In multiple linear regression, it is possible that some of the independent variables are actually correlated with one another, so it is important to check these before developing the regression model. If two independent variables are too highly correlated (r2 > ~0.6), then only one of them should be used in the regression model.
# 
# - Normality: The data follows a normal distribution.
# 
# - Linearity: the line of best fit through the data points is a straight line, rather than a curve or some sort of grouping factor.
# 
# In this practice project, you will fit the multiple linear regression parameters $(w,b)$ to your dataset. The model's prediction with multiple variables is given by the linear model: 
#     $$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}$$
#     or in vector notation:
#     $$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2} $$ 
#     where $\cdot$ is a vector `dot product`
#     
# 

# ## 2 Problem Statement
# 
# This is a classic example of Venture Capitalist Fund Challenge. The dataset contains data for various companies with four features (R&D Spend, Administration, Marketing Spend and, State) shown in the table below.  Note that, unlike the earlier labs, size is in sqft rather than 1000 sqft. This causes an issue, which you will solve in the next lab!
# 
# | R&D Spend       | Administration      | Marketing Spend  | State        | Profit        |   
# | ----------------| ------------------- |----------------- |--------------|-------------- |  
# | 165349.20       | 136897.80           | 471784.10        | New York     | 192261.83     |  
# | 162597.70       | 151377.59           | 443898.53        | California   | 191792.06     |  
# | 153441.51       | 101145.55           | 407934.54        | Florida      | 191050.39     |  
# 
# The challenge is to build a regression model using these values which can then predict the profit mergin for other companies with those exact input type feature data.

# ## Importing the libraries

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ## Importing the dataset

# In[8]:


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset
X
y


# ## Encoding categorical data

# In[9]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[10]:


print(X)


# ## Splitting the dataset into the Training set and Test set

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Training the Multiple Linear Regression model on the Training set

# In[12]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[13]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:





# In[ ]:




