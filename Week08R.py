#!/usr/bin/env python
# coding: utf-8

# # Multiple Regression (Continued)

# In[ ]:


from Week07R import *


# ## Standard Errors of Regression Coefficients

# We can take the same approach to estimating the standard errors of our regression
# coefficients. We repeatedly take a `bootstrap_sample` of our data and estimate `beta`
# based on that sample.
# 
# If the coefficient corresponding to one of the independent variables
# (say, `num_friends`) doesn’t vary much across samples, then we can be confident
# that our estimate is relatively tight.
# 
# If the coefficient varies greatly across samples,
# then we can’t be at all confident in our estimate.
# 
# The only subtlety is that, before sampling, we’ll need to `zip` our `x` data and `y` data to
# make sure that corresponding values of the independent and dependent variables are
# sampled together. This means that bootstrap_sample will return a list of pairs `(x_i,
# y_i)`, which we’ll need to reassemble into an `x_sample` and a `y_sample`:

# In[ ]:


from typing import Tuple

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]
    beta = least_squares_fit(x_sample, y_sample)
    print("bootstrap sample", beta)
    return beta

random.seed(0) # so that you get the same results when you run this code

# This might take a little while
bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
                                      estimate_sample_beta,
                                      100)


# After which we can estimate the standard deviation of each coefficient:

# In[ ]:


from statistics import mean

bootstrap_standard_errors = [stdev([beta[i] for beta in bootstrap_betas])
                             for i in range(4)]

bootstrap_param_means = [mean([beta[i] for beta in bootstrap_betas])
                             for i in range(4)]

print(bootstrap_param_means)
print(bootstrap_standard_errors)


# ```python
# # [0.846, # constant term, actual error = 1.19
# #  0.066, # num_friends,   actual error = 0.080
# #  0.107, # work_hours,    actual error = 0.127
# #  0.885] # phd,           actual error = 0.998
# ```
# 
# (We would likely get better estimates if we collected a lot more samples to estimate each `beta`, but we don’t have all day.)
# 
# If you have some background in statistics, you could also go ahead and try to perform a hypothesis test to see if each $\beta_i$ parameter equals to 0 or not. For which you may be able to compute a "p-value", which is the probablility value that has the following interpretation: if we assume that the true $\beta_i$ parameter is essentially 0, what is the probability of obtaining an actual $\beta_i$ this extreme from the sample?
# 
# Or put it simply, **if the p-value for a parameter lower than some signficance threshold (e.g., 0.05 is a common one), then we assume that parameter is actually significant** in predicting the response variable. Please keep this in mind when you use some regression software and it outputs the p-values for the parameters.

# ## Regularization

# In practice, you’d often like to apply linear regression to datasets with large numbers
# of variables. This creates a couple of extra wrinkles.
# - First, the more variables you use, the more likely you are to overfit your model to the training set.
# - Second, the more nonzero coefficients you have, the harder it is to make sense of them (If the goal is to explain some phenomenon, a sparse model with three factors might be more useful than a slightly better model with hundreds).
# 
# *Regularization* is an approach in which we add to the error term a penalty that gets
# larger as beta gets larger. We then minimize the combined error and penalty. The
# more importance we place on the penalty term, the more we discourage large coefficients.
# 
# For example, in *ridge regression*, we add a penalty proportional to the sum of the
# squares of the `beta_i` (except that typically we don’t penalize `beta_0`, the constant
# term):

# In[ ]:


# alpha is a *hyperparameter* controlling how harsh the penalty is
# sometimes it's called "lambda" but that already means something in Python
def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector,
                        y: float,
                        beta: Vector,
                        alpha: float) -> float:
    """estimate error plus ridge penalty on beta"""
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)

def sum_of_sqerrors_ridge(xs: List[Vector], ys: Vector, beta: Vector, alpha) -> float:
    return sum(squared_error_ridge(x_i, y_i, beta, alpha) for x_i, y_i in zip(xs, ys))


# We can then plug this into an optimizer in the usual way:

# In[ ]:


def least_squares_fit_ridge(xs: List[Vector], ys: Vector, alpha: float) -> Vector:
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta).
    """
    def object_fun (beta: Vector):
        return sum_of_sqerrors_ridge (xs, ys, beta, alpha)
    
    # Start with a random guess
    guess = [random.random() for _ in xs[0]]

    optim = minimize (object_fun, guess, method='BFGS')
    
    return optim.x


# With `alpha` set to 0, there’s no penalty at all and we get the same results as before:

# In[ ]:


random.seed(0)
beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0)

print(f"beta = {beta_0}")
print(f"sum of squared betas = {dot(beta_0[1:], beta_0[1:])}")
print(f"R^2 = {multiple_r_squared(inputs, daily_minutes_good, beta_0)}")


# As we increase `alpha`, the goodness of fit gets worse, but the size of `beta` gets smaller:

# In[ ]:


beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1)

print(f"beta = {beta_0_1}")
print(f"sum of squared betas = {dot(beta_0_1[1:], beta_0_1[1:])}")
print(f"R^2 = {multiple_r_squared(inputs, daily_minutes_good, beta_0_1)}")


# In[ ]:


beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1)

print(f"beta = {beta_1}")
print(f"sum of squared betas = {dot(beta_1[1:], beta_1[1:])}")
print(f"R^2 = {multiple_r_squared(inputs, daily_minutes_good, beta_1)}")


# In[ ]:


beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good, 10)

print(f"beta = {beta_10}")
print(f"sum of squared betas = {dot(beta_10[1:], beta_10[1:])}")
print(f"R^2 = {multiple_r_squared(inputs, daily_minutes_good, beta_10)}")


# In particular, the coefficient on “PhD” vanishes as we increase the penalty, which
# accords with our previous result that it wasn’t significantly different from 0.
# 
# Note: usually you’d want to rescale your data before using this
# approach. After all, if you changed years of experience to centuries
# of experience, its least squares coefficient would increase by a factor
# of 100 and suddenly get penalized much more, even though it’s
# the same model.
# 
# Another approach is *lasso regression*, which uses the penalty:

# In[ ]:


def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])


# Whereas the ridge penalty shrank the coefficients overall, the lasso penalty tends to
# force coefficients to be 0, which makes it good for learning sparse models.

# ## Linear Regression in scikit-learn

# Documentation Link: https://scikit-learn.org/stable/modules/linear_model.html
# 
# This example uses the only the first feature of the `diabetes` dataset, in order to illustrate a two-dimensional plot of this regression technique. The straight line can be seen in the plot, showing how linear regression attempts to draw a straight line that will best minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.

# In[ ]:


# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature at column 2, for illustrative purposes only so that it's easy to plot the graph later
# Note that in scikit-learn, there's no inherent distinction between Simple Linear Regression and Multiple Linear Regression
# Just pass in the desired X, which contains your features (1 feature: simple linear regression. n features: multiple regression)
# There's no need to include a column of 1's in your X here, it can be controlled by a parameter named "fit_intercept",
# when creating the LinearRegression object later.
diabetes_X = diabetes_X[:, 2:3]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The intercept:
print('Intercept: \n', regr.intercept_)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('R-Squared: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# And here's a super simple example to fit ridge regression:

# In[ ]:


reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0.2, 1], [1, 1]], [0, .1, 1])
print(reg.intercept_)
print(reg.coef_)


# And here's lasso regression:

# In[ ]:


reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [0.2, 1], [1, 1]], [0, .1, 1])
print(reg.intercept_)
print(reg.coef_)


# which can be compared with simple linear regression:

# In[ ]:


reg = linear_model.LinearRegression()
reg.fit([[0, 0], [0.2, 1], [1, 1]], [0, .1, 1])
print(reg.intercept_)
print(reg.coef_)


# You can also do polynomial regressions (e.g., quadratic regression, cubic regression, with or without any interaction terms) by expanding the design matrix x with new columns representing the polynomial powers or products of the original columns and then pass the new X to a linear regression object.

# In[ ]:


import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(20).reshape(10, 2)
print(X)

y = [(x_i[0] + x_i[1] + np.random.randn()) ** 2  for x_i in X]

poly = PolynomialFeatures(2, include_bias = False) # the initial columns of 1's not needed for LinearRegression()
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


# ## Linear Regression in statsmodels

# In[ ]:


import statsmodels.api as sm

spector_data = sm.datasets.spector.load()

# unlike scikit-learn, you are required to add a column of 1's if you want to fit the intercept
spector_data.exog1 = sm.add_constant(spector_data.exog) 

print(spector_data.endog) # endogeneous/response/dependent/y variable
print(spector_data.exog1)  # exogenous/explanatory/independent/x variable

# Fit and summarize Ordinary Least Squares (OLS) model
# Be sure to pass in y first, followed by X!
mod = sm.OLS(spector_data.endog, spector_data.exog1)
res = mod.fit()
print(res.summary())


# In[ ]:


# Quadratic Regression
# We will just borrow PolynomialFeatures from scikit-learn to generate the appropriate columns for polynomial regression
poly = PolynomialFeatures(2, include_bias = True)
spector_data.exog2 = poly.fit_transform (spector_data.exog)

print(spector_data.exog2)

mod2 = sm.OLS(spector_data.endog, spector_data.exog2)
res2 = mod2.fit()
print(res2.summary())


# In[ ]:


# Ridge Regression, call fit_regularized instead of fit, set L1_wt = 0
res_ridge = mod.fit_regularized(alpha = 0.05, L1_wt = 0)
print(res_ridge.params)


# In[ ]:


# Lasso Regression, call fit_regularized instead of fit, set L1_wt = 1
res_lasso = mod.fit_regularized(alpha = 0.05, L1_wt = 1) # Lasso
print(res_lasso.params)

