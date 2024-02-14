#!/usr/bin/env python
# coding: utf-8

# # Working with Data (Continued)

# In[ ]:


from Week05M import *


# ## Rescaling

# Many techniques are sensitive to the scale of your data. For example, imagine that you have a dataset consisting of the heights and weights of hundreds of data scientists, and that you are trying to identify clusters of body sizes.
# 
# Intuitively, we’d like clusters to represent points near each other, which means that we need some notion of distance between points. We could use the Euclidean distance, so a natural approach might be to treat (height, weight) pairs as points in two-dimensional space. Consider the people listed in the following table:
# 
# | Person | Height (in) | Height (cm) | Weight (lb) |
# |--------|-------------|-------------|-------------|
# | A      | 63          | 160         | 150         |
# | B      | 67          | 170.2       | 160         |
# | C      | 70          | 177.8       | 171         |
# 
# If we measure height in inches, then B’s nearest neighbor is A:

# In[ ]:


from scipy.spatial import distance

print(f"a to b = {distance.euclidean([63, 150], [67, 160])}")        # 10.77
print(f"a to c = {distance.euclidean([63, 150], [70, 171])}")        # 22.14
print(f"b to c = {distance.euclidean([67, 160], [70, 171])}")        # 11.40


# However, if we measure height in centimeters, then B’s nearest neighbor is instead C:

# In[ ]:


print(f"a to b = {distance.euclidean([160, 150], [170.2, 160])}")    # 14.28
print(f"a to c = {distance.euclidean([160, 150], [177.8, 171])}")    # 27.53
print(f"b to c = {distance.euclidean([170.2, 160], [177.8, 171])}")  # 13.37


# Obviously it’s a problem if changing units can change results like this. For this reason, when dimensions aren’t comparable with one another, we will sometimes rescale our data so that each dimension has mean 0 and standard deviation 1. This effectively gets rid of the units, converting each dimension to "standard deviations from the mean."
# 
# To start with, we’ll need to compute the mean and the standard_deviation for each position:

# In[ ]:


from typing import Tuple

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """returns the means and standard deviations for each position"""
    n = len(data)

    means  = [mean(data[i])  for i in range(n)]
    stdevs = [stdev(data[i]) for i in range(n)]

    return means, stdevs


# In[ ]:


vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)

print(f"means  = {means}")
print(f"stdevs = {stdevs}")


# We can then use them to create a new dataset:

# In[ ]:


def rescale(data: List[Vector]) -> List[Vector]:
    """
    Rescales the input data so that each position has
    mean 0 and standard deviation 1. (Leaves a position
    as is if its standard deviation is 0.)
    """
    means, stdevs = scale(data)

    # Make a copy of each vector
    rescaled = [v[:] for v in data]
    
    for i in range(len(data)):
        if stdevs[i] > 0:
            for j in range(len(data[i])):
                rescaled[i][j] = (rescaled[i][j] - means[i]) / stdevs[i]
                
    return rescaled


# Of course, let’s write a test to conform that rescale does what we think it should:

# In[ ]:


# after rescaling, we should have obtained vectors of mean 0 and stdev 1, except for the last vector
means, stdevs = scale(rescale(vectors))
print("After rescaling:")
print(f"means  = {means}")
print(f"stdevs = {stdevs}")


# ## An Aside: tqdm

# Frequently we’ll end up doing computations that take a long time. When you’re doing such work, you’d like to know that you’re making progress and how long you should expect to wait.
# 
# One way of doing this is with the tqdm library, which generates custom progress bars. We’ll use it some throughout the rest of the book, so let’s take this chance to learn how it works.
# 
# There are only a few features you need to know about. The first is that an iterable wrapped in tqdm.tqdm will produce a progress bar:

# In[ ]:


import tqdm
import random

for i in tqdm.tqdm(range(100)):
    # do something slow
    _ = [random.random() for _ in range(1000000)]


# In particular, it shows you what fraction of your loop is done (though it can’t do this if you use a generator), how long it’s been running, and how long it expects to run.
# 
# In this case (where we are just wrapping a call to `range`) you can just use `tqdm.trange`.
# 
# You can also set the description of the progress bar while it’s running. To do that, you need to capture the `tqdm` iterator in a `with` statement:

# In[ ]:


import time

def primes_up_to(n: int) -> List[int]:
    primes = [2]
    
    with tqdm.trange(3, n) as t:
        for i in t:
            # i is prime if no smaller prime divides it
            i_is_prime = not any(i % p == 0 for p in primes)
        
            if i_is_prime:
                primes.append(i)
        
            t.set_description(f"{len(primes)} primes")
            
            time.sleep(0.1) # manually sleep for 0.1 seconds per iteration to slow down the execution of our loop
    return primes

my_primes = primes_up_to(100)
print(my_primes)


# Using `tqdm` will occasionally make your code flaky—sometimes the screen redraws poorly, and sometimes the loop will simply hang. And if you accidentally wrap a tqdm loop inside another tqdm loop, strange things might happen. Typically its benefits outweigh these downsides, though, so we’ll try to use it whenever we have slowrunning computations.

# # Machine Learning

# Many people imagine that data science is mostly machine learning and that data scientists mostly build and train and tweak machine learning models all day long. (Then again, many of those people don’t actually know what machine learning is.)
# 
# In fact, data science is mostly turning business problems into data problems and collecting data and understanding data and cleaning data and formatting data, after which machine learning is almost an afterthought.

# ## Modeling

# Before we can talk about machine learning, we need to talk about *models*.
# 
# What is a model? It’s simply a specification of a mathematical (or probabilistic) relationship that exists between different variables. In simple terms, think of models as equations that takes in certain inputs and produce certain outputs.
# 
# For instance, if you’re trying to raise money for your social networking site, you might build a *business model* (likely in a spreadsheet) that takes inputs like "number of users," "ad revenue per user," and "number of employees" and outputs your annual
# profit for the next several years.
# 
# A cookbook recipe entails a model that relates inputs like "number of eaters" and "hungriness" to quantities of ingredients needed.
# 
# And if you’ve ever watched poker on television, you know that each player’s "win probability" is estimated in real time based on a model that takes into account the cards that have been revealed so far and the distribution of cards in the deck.

# ## What Is Machine Learning?

# Everyone has his or her own exact definition, but we’ll use *machine learning* to refer to creating and using models that are *learned from data*. In other contexts this might be called *predictive modeling* or *data mining*, but we will stick with machine learning.
# 
# Typically, our goal will be to use existing data to develop models that we can use to predict various outcomes for new data, such as:
# - Whether an email message is spam or not
# - Whether a credit card transaction is fraudulent
# - Which advertisement a shopper is most likely to click on
# - Which football team is going to win the Super Bowl
# 
# We’ll mainly be looking at supervised models (in which there is a set of data labeled with the correct answers to learn from) and unsupervised models (in which there are no such labels).
# 
# Now, in even the simplest situation there are entire universes of models that might describe the relationship we’re interested in. In most cases we will ourselves choose a *parameterized* family of models and then use data to learn parameters that are in
# some way optimal.
# 
# For instance, we might assume that a person’s height is (roughly) a linear function of his weight and then use data to learn what that linear function is (e.g., $height = m \times weight + b$, and we are trying to learn the best $m$ and $b$ from data).
# 
# Or we might assume that a decision tree is a good way to diagnose what diseases our patients have and then use data to learn the “optimal” such tree.
# 
# But before we can do that, we need to better understand the fundamentals of machine learning. We’ll discuss some of those basic concepts before we move on to the models themselves.

# ## Overfitting and Underfitting

# A common danger in machine learning is *overfitting*—producing a model that performs well on the data you train it on but generalizes poorly to any new data. This could involve learning *noise* in the data. Or it could involve learning to identify specific inputs rather than whatever factors are actually predictive for the desired output.
# 
# The other side of this is *underfitting*—producing a model that doesn’t perform well even on the training data, although typically when this happens you decide your model isn’t good enough and keep looking for a better one.
# 
# In the following figure, three polynomials are fit to a sample of data. (Don’t worry about how; we’ll get to that in later chapters.)

# ![Overfitting and Underfitting](fit.jpg)

# The horizontal line shows the best fit degree 0 (i.e., constant) polynomial, which is a horizontal line on the mean. It severely underfits the training data.
# 
# The best fit degree 9 (i.e., 10-parameter) polynomial goes through every training data point exactly, but it very severely overfits; if we were to pick a few more data points, it would quite likely miss them by a lot.
# 
# And the degree 1 line strikes a nice balance; it’s pretty close to every point, and—if these data are representative—the line will likely be close to new data points as well.
# 
# Clearly, models that are too complex lead to overfitting and don’t generalize well beyond the data they were trained on. So how do we make sure our models aren’t too complex? The most fundamental approach involves using different data to train the model and to test the model.
# 
# The simplest way to do this is to split the dataset, so that (for example) three-fourths of it is used to train the model, after which we measure the model’s performance on the remaining third:

# In[ ]:


import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')  # generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    random.shuffle(data)              # because shuffle modifies the list.
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)


# In[ ]:


# The proportions should be correct
print(len(train))
print(len(test))

# And the original data should be preserved (in some order)
assert sorted(train + test) == data


# Often, we’ll have paired input variables and output variables. In that case, we need to make sure to put corresponding values together in either the training data or the test data:

# In[ ]:


Y = TypeVar('Y')  # generic type to represent output variables

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # Generate the indices and split them.
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test


# As always, we want to make sure our code works right:

# In[ ]:


xs = [x for x in range(1000)]  # xs are 0 ... 999
ys = [2 * x for x in xs]       # each y_i is twice x_i
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

# Check that the proportions are correct
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

print("Training Set:")
print(x_train[:10])
print(y_train[:10])

print("Testing Set:")
print(x_test[:10])
print(y_test[:10])


# If the model was overfit to the training data, then it will hopefully perform really poorly on the (completely separate) test data. Said differently, if it performs well on the test data, then you can be more confident that it’s *fitting* rather than *overfitting*.

# ## Correctness

# Suppose I’ve come up with a cheap, noninvasive test that can be given to a newborn baby that predicts—with greater than 98% accuracy—whether the newborn will ever develop leukemia. So here's the test: *predict leukemia if and only if the baby is named Luke (which sounds sort of like "leukemia")*.
# 
# As we'll see, this test is indeed more than 98% accurate. Nonetheless, it's an incredibly stupid test, and a good illustration of why we don’t typically use "accuracy" to measure how good a (binary classification) model is.
# 
# Imagine building a model to make a binary judgment. Is this email spam? Should we hire this candidate?
# 
# Given a set of labeled data and such a predictive model, every data point lies in one of four categories:
# - *True positive*: "This message is spam, and we correctly predicted spam."
# - *False positive (Type 1 error)*: "This message is not spam, but we predicted spam."
# - *False negative (Type 2 error)*: "This message is spam, but we predicted not spam."
# - *True negative*: "This message is not spam, and we correctly predicted not spam."
# 
# We often represen these as counts in a *confusion matrix* (The name stems from the fact that the table makes it easy to see if the model is confusing two classes by mislabeling one as another):
# 
# |                    | Spam           | Not Spam       |
# |:-------------------|:---------------|:---------------|
# | Predict "Spam"     | True Positive  | False Positive |
# | Predict "Not Spam" | False Negative | True Negative  |
# 
# Let’s see how my leukemia test fits into this framework. These days approximately 5 babies out of 1,000 are named Luke. And the lifetime prevalence of leukemia is about 1.4%, or 14 out of every 1,000 people.
# 
# If we believe these two factors are independent and apply my "Luke is for leukemia" test to 1 million people, we’d expect to see a confusion matrix like:
# 
# |            | Leukemia | No Leukemia | Total     |
# |:-----------|----------|-------------|-----------|
# | "Luke"     | 70       | 4,930       | 5,000     |
# | Not "Luke" | 13,930   | 981,070     | 995,000   |
# | Total      | 14,000   | 986,000     | 1,000,000 |
# 
# We can then use these to compute various statistics about model performance. For example, *accuracy* is defined as the fraction of correct predictions:

# In[ ]:


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


# In[ ]:


print(f"accuracy = {accuracy(70, 4930, 13930, 981070)}")


# That seems like a pretty impressive number. But clearly this is not a good test, which means that we probably shouldn’t put a lot of credence in raw accuracy.
# 
# It’s common to look at the combination of *precision* and *recall*. Precision measures how accurate our *positive* predictions were:

# In[ ]:


def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)


# In[ ]:


print(f"precision = {precision(70, 4930, 13930, 981070)}")


# And recall measures what fraction of the positives our model identified:

# In[ ]:


def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)


# In[ ]:


print(f"recall = {recall(70, 4930, 13930, 981070)}")


# These are both terrible numbers, reflecting that this is a terrible model.
# 
# Sometimes precision and recall are combined into the *F1* score, which is defined as:

# In[ ]:


def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)


# In[ ]:


print(f"F1 score = {f1_score(70, 4930, 13930, 981070)}")


# This is the harmonic mean of precision and recall and necessarily lies between them.
# 
# Usually the choice of a model involves a tradeoff between precision and recall. A model that predicts "yes" when it’s even a little bit confident will probably have a high recall but a low precision; a model that predicts "yes" only when it’s extremely confident is likely to have a low recall and a high precision.
# 
# Alternatively, you can think of this as a tradeoff between false positives and false negatives. Saying "yes" too often will give you lots of false positives; saying "no" too often will give you lots of false negatives.
# 
# Imagine that there were 10 risk factors for leukemia, and that the more of them you had the more likely you were to develop leukemia. In that case you can imagine a continuum of tests: "predict leukemia if at least one risk factor," "predict leukemia if at least two risk factors," and so on.
# 
# As you increase the threshold, you increase the test’s precision (since people with more risk factors are more likely to develop the disease), and you decrease the test’s recall (since fewer and fewer of the eventual disease sufferers will meet the threshold). In cases like this, choosing the right threshold is a matter of finding the right tradeoff.
