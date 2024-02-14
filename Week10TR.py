#!/usr/bin/env python
# coding: utf-8

# # Gradient Descent

# Frequently when doing data science, we’ll be trying to the find the best model for a
# certain situation. And usually “best” will mean something like “minimizes the error
# of its predictions” In other words, it will
# represent the solution to some sort of optimization problem.
# 
# This means we’ll need to solve a number of optimization problems. Our approach will be a technique called gradient
# descent.

# ## The Idea Behind Gradient Descent

# Suppose we have some function `f` that takes as input a vector of real numbers and
# outputs a single real number. One simple such function is:

# In[1]:


import numpy as np
from typing import List
Vector = List[float]

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return np.dot(v, v)


# We’ll frequently need to maximize or minimize such functions. That is, we need to
# find the input `v` that produces the largest (or smallest) possible value.
# 
# For functions like ours, the gradient (which is the vector of partial derivatives) gives the input direction in which the function most quickly increases. (If you don’t know your calculus, take my word for it or look it up on
# the internet.)
# 
# Accordingly, one approach to maximizing a function is to pick a random starting
# point, compute the gradient, take a small step in the direction of the gradient (i.e., the
# direction that causes the function to increase the most), and repeat with the new
# starting point. Similarly, you can try to minimize a function by taking small steps in
# the opposite direction.
# 
# ![Finding a minimum using gradient descent](finding_min.jpg)
# 
# Note: If a function has a unique global minimum, this procedure is likely
# to find it. If a function has multiple (local) minima, this procedure
# might “find” the wrong one of them, in which case you might rerun
# the procedure from different starting points. If a function has no
# minimum, then it’s possible the procedure might go on forever.

# ## Estimating the Gradient

# If `f` is a function of one variable, its derivative at a point `x` measures how `f(x)`
# changes when we make a very small change to `x`. The derivative is defined as the limit
# of the difference quotients as `h`, the small change to `x`, approaches 0:

# In[ ]:


from typing import Callable

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h


# The derivative is the slope of the tangent line at `(x, f(x))` , while the difference quotient
# is the slope of the not-quite-tangent line that runs through `(x + h, f(x + h))` . As `h`
# gets smaller and smaller, the not-quite-tangent line gets closer and closer to the tangent
# line.
# 
# ![Approximating a derivative with a difference quotient](approx_derivative.jpg)

# For many functions it’s easy to exactly calculate derivatives. For example, the `square`
# function:

# In[3]:


def square(x: float) -> float:
    return x * x


# has the derivative:

# In[4]:


def derivative(x: float) -> float:
    return 2 * x


# which is easy for us to check by explicitly computing the difference quotient and taking
# the limit.
# 
# What if you couldn’t (or didn’t want to) find the gradient? Although we can’t take limits
# in Python, we can estimate derivatives by evaluating the difference quotient for a
# very small `h`

# In[5]:


xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

# plot to show they're basically the same
import matplotlib.pyplot as plt

plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label='Actual')     # red x
plt.plot(xs, estimates, 'b+', label='Estimate') # blue +
plt.legend(loc=9)
plt.show()


# When `f` is a function of many variables, it has multiple partial derivatives, each indicating
# how `f` changes when we make small changes in just one of the input variables.
# 
# We calculate (or estimate) its `i`th partial derivative by treating it as a function of just its `i`th variable,
# holding the other variables fixed:

# In[6]:


def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)    # add h to just the ith element of v
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


# after which we can estimate the gradient the same way:

# In[7]:


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]


# A major drawback to this “estimate using difference quotients”
# approach is that it’s computationally expensive. If `v` has length `n`,
# `estimate_gradient` has to evaluate `f` on `2n` different inputs. If
# you’re repeatedly estimating gradients, you’re doing a whole lot of
# extra work. Often times, we’ll use math to calculate our
# gradient functions explicitly.

# ## Using the Gradient

# It’s easy to see that the `sum_of_squares` function is smallest when its input `v` is a vector
# of zeros. But imagine we didn’t know that.
# 
# Let’s use gradients to find the minimum
# among all three-dimensional vectors. We’ll just pick a random starting point
# and then take tiny steps in the opposite direction of the gradient until we reach a
# point where the gradient is very small:

# In[8]:


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = np.multiply(step_size, gradient)
    return np.add(v, step)


# In[9]:


def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]


# In[10]:


import random

# pick a random starting point
v = [random.uniform(-10, 10) for i in range(3)]


# In[11]:


for epoch in range(1000):
    grad = sum_of_squares_gradient(v)    # compute the gradient at v
    v = gradient_step(v, grad, -0.01)    # take a negative gradient step
    print(epoch, v)


# In[12]:


from scipy.spatial import distance

assert distance.euclidean(v, [0, 0, 0]) < 0.001    # v should be close to 0


# If you run this, you’ll find that it always ends up with a `v` that’s very close to `[0,0,0]`.
# The more epochs you run it for, the closer it will get.

# ## Choosing the Right Step Size

# Although the rationale for moving against the gradient is clear, how far to move is
# not. Indeed, choosing the right step size is more of an art than a science. Popular
# options include:
# - Using a fixed step size
# - Gradually shrinking the step size over time
# - At each step, choosing the step size that minimizes the value of the objective function
# 
# The last approach sounds great but is, in practice, a costly computation.
# 
# To keep
# things simple, we’ll mostly just use a fixed step size. The step size that “works”
# depends on the problem—too small, and your gradient descent will take forever; too
# big, and you’ll take giant steps that might make the function you care about get larger
# or even be undefined. So we’ll need to experiment.

# ## Using Gradient Descent to Fit Models

# We’ll be using gradient descent to fit parameterized models to data.
# 
# In
# the usual case, we’ll have some dataset and some (hypothesized) model for the data
# that depends (in a differentiable way) on one or more parameters.
# 
# We’ll also have a
# loss function that measures how well the model fits our data. (e.g., sse, smaller is better.)
# 
# If we think of our data as being fixed, then our loss function tells us how good or bad
# any particular model parameters are. This means we can use gradient descent to find
# the model parameters that *make the loss as small as possible*.
# 
# Let’s look at a simple
# example:

# In[14]:


# x ranges from -50 to 49, y is always 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

print(inputs)


# In this case we know the parameters of the linear relationship between `x` and `y`, but
# imagine we’d like to learn them from the data. We’ll use gradient descent to find the
# slope and intercept that minimize the average squared error.
# 
# We’ll start off with a function that determines the gradient based on the error from a
# single data point.
# 
# $$ y = mx + b + \epsilon = \hat{y} + \epsilon$$
# 
# $$ \epsilon = y - \hat{y} $$
# 
# We then need a function that takes in our parameters as input and returns the squared error as output and we would be interested in minimizing this function. Let's call this function $l$.
# 
# $$ l(m, b) = \epsilon^2 = (y - \hat{y})^2 = (y - mx - b)^2 $$
# 
# Now let's take the gradient of $l$.
# 
# $$ \frac{\delta l}{\delta m} = 2 * (y - mx - b) * (-x) = -2 * \epsilon * x $$
# 
# $$ \frac{\delta l}{\delta b} = 2 * (y - mx - b) * (-1) = -2 * \epsilon$$

# In[15]:


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept      # The prediction of the model.
    error = y - predicted                  # error is (actual - predicted)
    squared_error = error ** 2             # We'll minimize squared error
    grad = [-2 * error * x, -2 * error]    # using its gradient.
    return grad


# Now, that computation was for a single data point. For the whole dataset we’ll look at
# the *mean squared error*. And the gradient of the mean squared error is just the mean
# of the individual gradients.
# 
# So, here’s what we’re going to do:
# 1. Start with a random value for theta.
# 2. Compute the mean of the gradients.
# 3. Adjust theta in the direction of negative gradient (theta = theta - learning_rate * gradient)
# 4. Repeat.
# 
# After a lot of epochs (what we call each pass through the dataset), we should learn
# something like the correct parameters:

# In[30]:


# Start with random values for slope and intercept.
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001

for epoch in range(5000):
    # Compute the mean of the gradients
    grad = np.mean([linear_gradient(x, y, theta) for x, y in inputs], axis = 0)
    # Take a step in that direction of negative gradient to minimize
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)


# In[17]:


slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"


# Note that the procedure is extremely sensitive to your choice the the `learning_rate` parameter.
# - too small, the process can be really slow
# - too big, the procedure may encounter numerical difficulties

# ## Minibatch and Stochastic Gradient Descent

# One drawback of the preceding approach is that we had to evaluate the gradients on
# the entire dataset before we could take a gradient step and update our parameters.
# In
# this case it was fine, because our dataset was only 100 pairs and the gradient computation
# was cheap.
# 
# Your models, however, will frequently have large datasets and expensive gradient
# computations. In that case you’ll want to take gradient steps more often.
# 
# We can do this using a technique called *minibatch gradient descent*, in which we compute
# the gradient (and take a gradient step) based on a “minibatch” sampled from the
# larger dataset:

# In[29]:


from typing import TypeVar, List, Iterator

T = TypeVar('T')  # this allows us to type "generic" functions

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # Start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


# Note: The `TypeVar(T)` allows us to create a “generic” function. It says that
# our dataset can be a list of any single type—`str`s, `int`s, `list`s,
# whatever—but whatever that type is, the outputs will be batches of
# it.
# 
# Now we can solve our problem again using minibatches:

# In[31]:


# Minibatch gradient descent example

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = np.mean([linear_gradient(x, y, theta) for x, y in batch], axis = 0)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)


# In[32]:


slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"


# Another variation is *stochastic gradient descent*, in which you take gradient steps
# based on one training example at a time:

# In[33]:


# Stochastic gradient descent example

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)


# In[34]:


slope, intercept = theta
assert 19.9 < slope < 20.1,   "slope should be about 20"
assert 4.9 < intercept < 5.1, "intercept should be about 5"


# On this problem, stochastic gradient descent finds the optimal parameters in a much
# smaller number of epochs. But there are always tradeoffs.
# 
# Basing gradient steps on
# small minibatches (or on single data points) allows you to take more of them, but the
# gradient for a single point might lie in a very different direction from the gradient for
# the dataset as a whole.
# 
# Note: The terminology for the various flavors of gradient descent is not
# uniform. The “compute the gradient for the whole dataset”
# approach is often called *batch gradient descent*, and some people
# say *stochastic gradient descent* when referring to the minibatch version
# (of which the one-point-at-a-time version is a special case).
