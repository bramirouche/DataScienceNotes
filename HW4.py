#!/usr/bin/env python
# coding: utf-8

# # CSCI 3360 Homework 4 - Brandon Amirouche

# ## Instructions

# consider the concrete compressive strength dataset (https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength). I have already separated the training and testing sets for you, so please directly work with \concretetraining.csv" and \concretetesting.csv" downloaded from eLC and answer the following questions.
# 
# Note:  you are strongly encouraged to use regression models from either scikit-learn or statsmodels.
# 
# 1.	Using the training set, plot histograms of the eight input features/variables as well as the response variable, which represents concrete compressive strength.
# 
# 
# 2.	Using the training set, obtain the mean and standard deviation of each input feature as well as the response variable.
# 
# 
# 3.	Using the training set, report the correlation between each input feature and the response variable.
# 
# 
# 4.	Based  on  the correlation values from the previous question, which three input features exhibit the greatest correlation with the response variable?  Create scatter plots of the response variable vs. each of the three input features.
# 
# 
# 5.	Train the multiple linear regression  model using the training set, predict on  both the training and testing set, and report the R2 values on both the training and testing set.
# 
# 
# 6.	Train the ridge regression model using the training set. Train multiple times  by trying different values and observe how the coefficients and the R2 values change. Are there any coefficients that are rapidly moving towards (or becoming) 0 as you increase? Do not go beyond a value that causes significant reduction in R2.
# 
# 
# 7.	Train the lasso regression model using the training set. Train multiple times  by  trying  different values and observe how the coefficients and the R2 values change. Are there any coefficients that are rapidly moving towards (or becoming) 0 as you increase? Do not go beyond an value that causes significant reduction in R2.
# 
# 
# 8.	Train the quadratic regression model (including the interaction terms) using the training set. Report the R2 values on both the training and testing set.
# 
# 
# 9.	Train the cubic regression model (including the interaction terms) using the training set. Report the R2 values on both the training and testing set.
# 
# 
# 10.	Based on everything that you have observed so far, which is the best model?  Please justify your answer.
# 

# ## Academic Honesty Statement

# In[1]:


# first fill in your name
first_name = "Brandon"
last_name  = "Amirouche"

print("****************************************************************")
print("CSCI 3360 Homework 4")
print(f"completed by {first_name} {last_name}")
print(f"""
I, {first_name} {last_name}, certify that the following code
represents my own work. I have neither received nor given 
inappropriate assistance. I have not consulted with another
individual, other than a member of the teaching staff, regarding
the specific problems in this homework. I recognize that any 
unauthorized assistance or plagiarism will be handled in 
accordance with the University of Georgia's Academic Honesty
Policy and the policies of this course.
""")
print("****************************************************************")


# ## Problem 1

# Using the training set, plot histograms of the eight input features/variables as well as the response variable, which represents concrete compressive strength.

# In[2]:


from typing import List
from collections import Counter
from typing import Dict
import csv
from collections import defaultdict
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import NamedTuple
from scipy.spatial import distance
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

header_list = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 
               'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Concrete compressive strength']
data = pd.read_csv('concrete_training.csv', names=header_list)
df = pd.DataFrame(data)
print(df)
print(df.hist())


# ## Problem 2

# Using the training set, obtain the mean and standard deviation of each input feature as well as the response variable.

# In[3]:


print(f"(a)\n")
#Mean by Column
print(f"\tThe Mean value of each input feature in the Training Set:")
print(f"\t\tThe mean of the Cement input feature = {df.mean()[0]}")
print(f"\t\tThe mean of the Blast Furnace Slag input feature = {df.mean()[1]}")
print(f"\t\tThe mean of the Fly Ash input feature = {df.mean()[2]}")
print(f"\t\tThe mean of the Water input feature = {df.mean()[3]}")
print(f"\t\tThe mean of the Superplasticizer input feature = {df.mean()[4]}")
print(f"\t\tThe mean of the Coarse Aggregate input feature = {df.mean()[5]}")
print(f"\t\tThe mean of the Fine Aggregate input feature = {df.mean()[6]}")
print(f"\t\tThe mean of the Age input feature = {df.mean()[7]}")
print(f"\t\tThe mean of the Concrete compressive strength response variable = {df.mean()[8]}")


print(f"(b)\n")
#Standard Deviation by Column
print(f"\tThe Standard Deviation value of each input feature in the Training Set:")
print(f"\t\tThe standard deviation of the Cement input feature = {df.std(axis=0)[0]}")
print(f"\t\tThe standard deviation of the Blast Furnace Slag input feature = {df.std(axis=0)[1]}")
print(f"\t\tThe standard deviation of the Fly Ash input feature = {df.std(axis=0)[2]}")
print(f"\t\tThe standard deviation of the Water input feature = {df.std(axis=0)[3]}")
print(f"\t\tThe standard deviation of the Superplasticizer input feature = {df.std(axis=0)[4]}")
print(f"\t\tThe standard deviation of the Coarse Aggregate input feature = {df.std(axis=0)[5]}")
print(f"\t\tThe standard deviation of the Fine Aggregate input feature = {df.std(axis=0)[6]}")
print(f"\t\tThe standard deviation of the Age input feature = {df.std(axis=0)[7]}")
print(f"\t\tThe standard deviation of the Concrete compressive strength response variable = {df.std(axis=0)[8]}")
print(f"\n")


# ## Problem 3

# Using the training set, report the correlation between each input feature and the response variable.

# In[4]:


#Correlation between each input feature and the response variable
print(f"\tThe correlation between each input feature and the response variable in the Training Set:")
(df.corr().abs())


# ## Problem 4

# Based  on  the correlation values from the previous question, which three input features exhibit the greatest correlation with the response variable?  Create scatter plots of the response variable vs. each of the three input features.

# In[5]:


print(f"\tScatter Plot for Concrete Compressive Strength(response variable) vs Cement(input variable):\n")
print(sns.scatterplot(x = 'Cement', y = 'Concrete compressive strength', data = df, color = 'r'))


# In[6]:


print(f"\tScatter Plot for Concrete Compressive Strength(response variable) vs Superplasticizer(input variable):\n")
print(sns.scatterplot(x = 'Superplasticizer', y = 'Concrete compressive strength', data = df, color = 'g'))


# In[7]:


print(f"\tScatter Plot for Concrete Compressive Strength(response variable) vs Age(input variable):\n")
print(sns.scatterplot(x = 'Age', y = 'Concrete compressive strength', data = df, color = 'c'))


# ## Problem 5

# Train the multiple linear regression  model using the training set, predict on  both the training and testing set, and report the R2 values on both the training and testing set.

# In[8]:


from typing import Tuple
from statistics import mean, stdev
from scipy.stats import pearsonr
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data1 = pd.read_csv('concrete_testing.csv', names=header_list)
df1 = pd.DataFrame(data1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training se
regr.fit(df[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']], df['Concrete compressive strength'])

# Make predictions using the training set
concrete_prediction = regr.predict(df[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']])
#R2 Value of the training set
print(f"R2 Value of the training set:\n")
print('\tR-Squared: %.2f'
      % r2_score(df['Concrete compressive strength'], concrete_prediction))

print(f"\n")

# Make predictions using the testing set
concrete_prediction = regr.predict(df1[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']])
#R2 Value of the testing set
print(f"R2 Value of the testing set:\n")
print('\tR-Squared: %.2f'
      % r2_score(df1['Concrete compressive strength'], concrete_prediction))


# ## Problem 6

# Train the ridge regression model using the training set. Train multiple times  by trying different values and observe how the coefficients and the R2 values change. Are there any coefficients that are rapidly moving towards (or becoming) 0 as you increase? Do not go beyond a value that causes significant reduction in R2.

# In[9]:


import statsmodels.api as sm
mod = sm.OLS(df['Concrete compressive strength'], df[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']])
res = mod.fit()

print(f"Ridge Regression when Alpha is .3:\n")
reg = linear_model.Ridge(alpha=.3)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Ridge Regression when Alpha is .5:\n")
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Ridge Regression when Alpha is .7:\n")
reg = linear_model.Ridge(alpha=.7)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Ridge Regression when Alpha is .9:\n")
reg = linear_model.Ridge(alpha=.9)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Ridge Regression when Alpha is .05:\n")
# alpha is .05
res_ridge = mod.fit_regularized(alpha = 0.05, L1_wt = 0)
print(res_ridge.params)
print(f"\n")

print(f"Ridge Regression when Alpha is .03:\n")
# alpha is .03
res_ridge = mod.fit_regularized(alpha = 0.03, L1_wt = 0)
print(res_ridge.params)
print(f"\n")

print(f"Ridge Regression when Alpha is .07:\n")
# alpha is .07
res_ridge = mod.fit_regularized(alpha = 0.07, L1_wt = 0)
print(res_ridge.params)
print(f"\n")
print(f"\n")

print(f"Both the first and second coefficient seems to be moving towards 0 as alpha increases.\n")


# ## Problem 7

# Train the lasso regression model using the training set. Train multiple times  by  trying  different values and observe how the coefficients and the R2 values change. Are there any coefficients that are rapidly moving towards (or becoming) 0 as you increase? Do not go beyond an value that causes significant reduction in R2.

# In[10]:


print(f"Lasso Regression when Alpha is .3:\n")
reg = linear_model.Lasso(alpha=0.3)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Lasso Regression when Alpha is .5:\n")
reg = linear_model.Lasso(alpha=0.5)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Lasso Regression when Alpha is .7:\n")
reg = linear_model.Lasso(alpha=0.7)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Lasso Regression when Alpha is .9:\n")
reg = linear_model.Lasso(alpha=0.9)
reg.fit([[0, 0], [0.5, 3], [3, 2]], [0, .6, 7])
print(reg.intercept_)
print(reg.coef_)
print(f"\n")

print(f"Lasso Regression when Alpha is .07:\n")
res_lasso = mod.fit_regularized(alpha = 0.05, L1_wt = 1) # Lasso
print(res_lasso.params)
print(f"\n")

print(f"Lasso Regression when Alpha is .03:\n")
# alpha is .03
res_lasso = mod.fit_regularized(alpha = 0.03, L1_wt = 1) # Lasso
print(res_lasso.params)
print(f"\n")

print(f"Lasso Regression when Alpha is .07:\n")
# alpha is .07
res_lasso = mod.fit_regularized(alpha = 0.07, L1_wt = 1) # Lasso
print(res_lasso.params)
print(f"\n")
print(f"\n")

print(f"The first coefficient is slowly approaching zero as alpha increases.\n") 
print(f"The second coefficient seems to be remaining at 0 regardless of alpha.\n")


# ## Problem 8

# Train the quadratic regression model (including the interaction terms) using the training set. Report the R2 values on both the training and testing set.

# In[11]:


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

X = np.arange(20).reshape(10, 2)
print(X)

y = [(x_i[0] + x_i[1] + np.random.randn()) ** 2  for x_i in X]

poly = PolynomialFeatures(2, include_bias = False)
X2 = poly.fit_transform(X)

print(X2)

reg = linear_model.LinearRegression()
reg.fit(X, y)
print(f"intercept = {reg.intercept_}")
print(f"coefficients = {reg.coef_}")
print(f"Linear Regression R-Squared = {r2_score(y, reg.predict(X))}")

reg.fit(X2, y)
print(f"intercept = {reg.intercept_}")
print(f"coefficients = {reg.coef_}")
print(f"Quadratic Regression R-Squared = {r2_score(y, reg.predict(X2))}")

print(f"\n")
print(f"\n")

# Summary of the training set
print(f"Summary of the training set including R2:\n")
mod = sm.OLS(df['Concrete compressive strength'], df[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']])
res = mod.fit()
print(res.summary())

print(f"\n")
print(f"\n")

# Summary of the testing set
print(f"Summary of the testing set including R2:\n")
mod1 = sm.OLS(df1['Concrete compressive strength'], df1[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']])
res1 = mod1.fit()
print(res1.summary())


# ## Problem 9

# Train the cubic regression model (including the interaction terms) using the training set. Report the R2 values on both the training and testing set.

# In[12]:


X = np.arange(20).reshape(10, 2)
print(X)

y = [(x_i[0] + x_i[1] + np.random.randn()) ** 3  for x_i in X]

poly1 = PolynomialFeatures(3, include_bias = False)
X3 = poly1.fit_transform(X)

print(X3)

reg = linear_model.LinearRegression()
reg.fit(X, y)
print(f"intercept = {reg.intercept_}")
print(f"coefficients = {reg.coef_}")
print(f"Linear Regression R-Squared = {r2_score(y, reg.predict(X))}")

reg.fit(X3, y)
print(f"intercept = {reg.intercept_}")
print(f"coefficients = {reg.coef_}")
print(f"Quadratic Regression R-Squared = {r2_score(y, reg.predict(X3))}")

print(f"\n")
print(f"\n")

# Summary of the training set
print(f"Summary of the training set including R2:\n")
mod = sm.OLS(df['Concrete compressive strength'], df[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']])
res = mod.fit()
print(res.summary())

print(f"\n")
print(f"\n")

# Summary of the testing set
print(f"Summary of the testing set including R2:\n")
mod1 = sm.OLS(df1['Concrete compressive strength'], df1[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']])
res1 = mod1.fit()
print(res1.summary())


# ## Problem 10

# Based on everything that you have observed so far, which is the best model?  Please justify your answer.

# Based on all of the information I have observed and tests ran, I believe the best model is the quadratic regression model. This is because it retains a high R2 number with a changing alpha, and doesn't seem to have the significant drop off. Also, the coefficients tend to approach zero at a more consistent rate.
