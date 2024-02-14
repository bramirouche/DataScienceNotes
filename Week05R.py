#!/usr/bin/env python
# coding: utf-8

# # Machine Learning (Continued)

# ## The Bias-Variance Tradeoff

# Another way of thinking about the overfitting problem is as a tradeoff between *bias* and *variance*. Both are measures of what would happen if you were to retrain your model many times on different sets of training data (from the same larger population).
# 
# Recall the figure illustrating how models may overfit or underfit data:

# ![Overfitting and Underfitting](fit.jpg)

# The degree 0 model will make a lot of mistakes for pretty much any training set (drawn from the same population), which means that it has a high *bias*. However, any two randomly chosen training sets should give pretty similar models (since any two randomly chosen training sets should have pretty similar average values). So we say that it has a low variance.
# 
# *High bias and low variance typically correspond to underfitting*.
# 
# On the other hand, the degree 9 model fit the training set perfectly. It has very low bias but very high variance (since any two training sets would likely give rise to very different models).
# 
# *Low bias and high variance typically correspond to overfitting*.
# 
# Thinking about model problems this way can help you figure out what to do when your model doesn’t work so well.
# 
# If your model has high bias (which means it performs poorly even on your training data), one thing to try is adding more features/inputs. Going from the degree 0 model to the degree 1 model was a big improvement.
# 
# If your model has high variance, you can similarly *remove* features. But another solution is to obtain more data (if you can).
# 
# The following figure illustrates how various degree 9 polynomials are fit to sample data of different sizes.

# ![Overfitting and Underfitting](fit2.jpg)

# The model fit based on 10 data points is all over the place, as we saw before.
# 
# If we instead train on 100 data points, there’s much less overfitting.
# 
# And the model trained from 1,000 data points looks very similar to the degree 1 model.
# 
# Holding model complexity constant, the more data you have, the harder it is to overfit. On the other hand, more data won’t
# help with bias. If your model doesn’t use enough features to capture regularities in the data, throwing more data at it won’t help.

# ## Feature Extraction and Selection

# As has been mentioned, when your data doesn’t have enough features, your model is likely to underfit. And when your data has too many features, it’s easy to overfit. But what are features, and where do they come from?
# 
# *Features* are whatever inputs we provide to our model.
# 
# In the simplest case, features are simply given to you. If you want to predict someone's salary based on her years of experience, then years of experience is the only feature you have.
# 
# Things become more interesting as your data becomes more complicated. Imagine trying to build a spam filter to predict whether an email is junk or not. Most models won’t know what to do with a raw email, which is just a collection of text. You’ll have
# to extract features. For example:
# 
# - Does the email contain the word *Nigeria* and the $ character?
# - How many times does the letter *d* appear?
# - What was the domain of the sender?
# 
# The answer to a question like the first question here is simply a yes or no, which we typically encode as a 1 or 0. The second is a number. And the third is a choice from a discrete set of options.
# 
# Pretty much always, we’ll extract features from our data that fall into one of these three categories. What’s more, the types of features we have constrain the types of models we can use.
# - The Naive Bayes classifier is suited to yes-or-no features, like the first one in the preceding list.
# - Regression models require numeric features (which could include dummy variables that are 0s and 1s).
# - decision trees can deal with numeric or categorical data.
# 
# Although in the spam filter example we looked for ways to create features, sometimes we’ll instead look for ways to remove features.
# 
# For example, your inputs might be vectors of several hundred numbers. Depending on the situation, it might be appropriate to distill these down to a handful of important dimensions and use only that small number of features. Or it might be appropriate to use a technique (e.g., regularization) that penalizes models the more features they use.
# 
# How do we choose features? That’s where a combination of experience and domain expertise comes into play. If you’ve received lots of emails, then you probably have a sense that the presence of certain words might be a good indicator of spamminess. And you might also get the sense that the number of *d*s is likely not a good indicator of spamminess. But in general this can be a trial and error process, or if feasible, do some sort of exhausive search on all possible subsets of the features.

# # k-Nearest Neighbors

# Imagine that you’re trying to predict how I’m going to vote in the next presidential election. If you know nothing else about me (and if you have the data), one sensible approach is to look at how my neighbors are planning to vote. Living in Clarke County, most of my neighbors tend to vote for the Democratic candidate, which suggests that "Democratic candidate" is a good guess for me as well.
# 
# Now imagine you know more about me than just geography—perhaps you know my age, my income, ethnicity, and so on. To the extent my behavior is influenced (or characterized) by those things, looking just at my neighbors who are close to me among all those dimensions seems likely to be an even better predictor than looking at all my neighbors. This is the idea behind *nearest neighbors classification*.

# ## The Model

# Nearest neighbors is one of the simplest predictive models there is. It makes no mathematical assumptions, and it doesn’t require any sort of heavy machinery. The only things it requires are:
# 
# - Some notion of distance
# - An assumption that points that are close to one another are similar
# 
# Most of the techniques we’ll see this semester look at the dataset as a whole in order to learn patterns in the data. Nearest neighbors, on the other hand, quite consciously neglects a lot of information, since the prediction for each new point depends only on the handful of points closest to it.
# 
# What’s more, nearest neighbors is probably not going to help you understand the drivers of whatever phenomenon you’re looking at. Predicting my votes based on my neighbors’ votes doesn’t tell you much about what causes me to vote the way I do, whereas some alternative model that predicted my vote based on (say) my income and marital status very well might.
# 
# In the general situation, we have some data points and we have a corresponding set of labels. The labels could be True and False, indicating whether each input satisfies some condition like “is spam?” or “is poisonous?” or “would be enjoyable to watch?” Or they could be categories, like movie ratings (G, PG, PG-13, R, NC-17). Or they could be the names of presidential candidates. Or they could be favorite programming languages.
# 
# In our case, the data points will be vectors, and we could make use of the Euclidean distance to determine the distance between two data points (vectors).
# 
# Let’s say we’ve picked a number *k* like 3 or 5. Then, when we want to classify some new data point, we find the k nearest labeled points and let them vote on the new output.
# 
# To do this, we’ll need a function that counts votes. One possibility is:

# In[ ]:


from typing import List
from collections import Counter

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner


# In[ ]:


print(raw_majority_vote(['a', 'b', 'c', 'b']))


# But this doesn’t do anything intelligent with ties. For example, imagine we’re rating
# movies and the five nearest movies are rated G, G, PG, PG, and R. Then G has two
# votes and PG also has two votes. In that case, we have several options:
# - Pick one of the winners at random.
# - Weight the votes by distance and pick the weighted winner.
# - Reduce k until we find a unique winner.
# 
# We’ll implement the third:

# In[ ]:


def majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to farthest."""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest


# In[ ]:


# Tie, so look at first 4, then 'b'
print(majority_vote(['a', 'b', 'c', 'b', 'a']))


# This approach is sure to work eventually, since in the worst case we go all the way
# down to just one label, at which point that one label wins.
# 
# With this function it’s easy to create a classifier:

# In[ ]:


from typing import NamedTuple
from scipy.spatial import distance
Vector = List[float]

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:

    # Order the labeled points from nearest to farthest.
    # the key parameter is used to specify a funciton whose return value is to be used as the key for sorting
    by_distance = sorted(labeled_points,
                         key=lambda lp: distance.euclidean(lp.point, new_point))

    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # and let them vote.
    return majority_vote(k_nearest_labels)


# Let’s take a look at how this works.

# ## Example: The Iris Dataset

# The *Iris* dataset is a well known dataset in machine learning. It contains a bunch of measurements for 150 flowers representing three species of iris. For each flower we have its petal
# length, petal width, sepal length, and sepal width, as well as its species. You can download
# it from https://archive.ics.uci.edu/ml/datasets/iris:

# In[ ]:


import requests

data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

with open('iris.data', 'w') as f:
    f.write(data.text)


# The data is comma-separated, with fields:
# 
# `sepal_length, sepal_width, petal_length, petal_width, class`
# 
# For example, the first row looks like:
# 
# `5.1,3.5,1.4,0.2,Iris-setosa`

# In this section we’ll try to build a model that can predict the class (that is, the species) from the first four measurements.
# 
# To start with, let’s load and explore the data. Our nearest neighbors function expects a
# `LabeledPoint`, so let’s represent our data that way:

# In[ ]:


from typing import Dict
import csv
from collections import defaultdict

def parse_iris_row(row: List[str]) -> LabeledPoint:
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    measurements = [float(value) for value in row[:-1]]
    # class is e.g. "Iris-virginica"; we just want "virginica"
    label = row[-1].split("-")[-1]

    return LabeledPoint(measurements, label)

with open('iris.data') as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader if row] # if row ensures that it's not empty
    
# We'll also group just the points by species/label so we can plot them.
points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

