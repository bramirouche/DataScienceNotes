#!/usr/bin/env python
# coding: utf-8

# # Decision Trees

# The VP of Talent at your social network start-up has interviewed a number of job candidates from the
# site, with varying degrees of success. He’s collected a dataset consisting of several
# (qualitative) attributes of each candidate, as well as whether that candidate interviewed
# well or poorly. Could you, he asks, use this data to build a model identifying
# which candidates will interview well, so that he doesn’t have to waste time conducting
# interviews?
# This seems like a good fit for a decision tree, another predictive modeling tool in the
# data scientist’s kit.
# 
# This seems like a good fit for a *decision tree*, another predictive modeling tool in the
# data scientist’s kit.

# ## What Is a Decision Tree?

# A decision tree uses a tree structure to represent a number of possible decision paths
# and an outcome for each path.
# 
# If you have ever played the game *Twenty Questions*, then you are familiar with decision
# trees. For example:
# - “I am thinking of an animal.”
# - “Does it have four legs?”
# - “No.”
# - “Does it fly?”
# - “Yes.”
# - “Does it appear on the back of a quarter coin?”
# - “Yes.”
# - “Is it an eagle?”
# - “Yes, it is!”
# 
# This corresponds to the path: "Does not have four legs" -> "Does fly" -> "On the quarter coin" -> "Eagle"
# 
# in a possible “guess the animal” decision tree. Below is an example of a (not so comprehensive) decision tree.
# 
# ![A "guess the animal" decision tree](decision_tree.jpg)
# 
# Decision trees have a lot to recommend them. They’re very easy to understand and
# interpret, and the process by which they reach a prediction is completely transparent.
# Unlike the other models we’ve looked at so far, decision trees can easily handle a mix
# of numeric (e.g., number of legs) and categorical (e.g., fly/not fly)
# attributes.
# 
# At the same time, finding an “optimal” decision tree for a set of training data is computationally
# a very hard problem. (We will get around this by trying to build a good enough
# tree rather than an optimal one, although for large datasets this can still be a
# lot of work.) More important, it is very easy (and very bad) to build decision trees
# that are overfitted to the training data, and that don’t generalize well to unseen data.
# We’ll look at ways to address this.
# 
# Most people divide decision trees into classification trees (which produce categorical
# outputs) and regression trees (which produce numeric outputs). Here, we’ll
# focus on classification trees, and we’ll work through the ID3 algorithm for learning a
# decision tree from a set of labeled data, which should help us understand how decision
# trees actually work.
# 
# To make things simple, we’ll restrict ourselves to problems
# with binary outputs like “Should I hire this candidate?” or “Should I show this website
# visitor advertisement A or advertisement B?” or “Will eating this food I found in
# the office fridge make me sick?”

# ## Entropy

# In order to build a decision tree, we will need to decide what questions to ask and in
# what order. At each stage of the tree there are some possibilities we’ve eliminated and
# some that we haven’t.
# 
# After learning that an animal doesn’t have more than five legs,
# we’ve eliminated the possibility that it’s a grasshopper. We haven’t eliminated the possibility
# that it’s a duck.
# 
# Each possible question *partitions* the remaining possibilities
# according to its answer.
# 
# Ideally, we’d like to choose questions whose answers give a lot of information about
# what our tree should predict.
# - If there’s a single yes/no question for which “yes” answers always correspond to True outputs and “no” answers to False outputs (or vice versa), this would be an awesome question to pick.
# - Conversely, a yes/no question for which neither answer gives you much new information about what the prediction should be is probably not a good choice.
# 
# We capture this notion of “how much information” with *entropy*. You have probably
# heard this term used to mean disorder. We use it to represent the uncertainty associated
# with data.
# 
# Imagine that we have a set $S$ of data, each member of which is labeled as belonging to
# one of a finite number of classes $C_1, \dots, C_n$.
# - If all the data points belong to a single class, then there is no real uncertainty, which means we’d like there to be low (or zero) entropy.
# - If the data points are evenly spread across the classes, there is a lot of uncertainty and we’d like there to be high entropy.
# 
# In math terms, if $p_i$ is the proportion of data labeled as class $C_i$, we define the entropy
# as:
# 
# $$ H(S) = -p_1\log_2 p_1 - \dots - p_n\log_2 p_n$$
# 
# with the (standard) convention that $0 \log 0 = 0$.
# 
# Without worrying too much about the grisly details, each term $−p_i \log_2 p_i$ is nonnegative
# and is close to 0 precisely when $p_i$ is either close to 0 or close to 1.
# 
# ![A graph of -p log p](entropy.jpg)
# 
# This means the entropy will be small when every $p_i$ is close to 0 or 1 (i.e., when most
# of the data is in a single class), and it will be larger when many of the $p_i$’s are not close
# to 0 (i.e., when the data is spread across multiple classes). This is exactly the behavior
# we desire.
# 
# It is easy enough to roll all of this into a function:

# In[1]:


from typing import List
import math

def entropy(class_probabilities: List[float]) -> float:
    """Given a list of class probabilities, compute the entropy"""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p > 0) # ignore zero probabilities since log(0) is undefined, but we will treat 0 log(0) as 0 by convention


# In[3]:


print(entropy([1.0]))
print(entropy([0.5, 0.5]))
print(entropy([0.25, 0.75]))
print(entropy([0.75, 0.25]))
print(entropy([0.95, 0.05]))


# Our data will consist of pairs `(input, label)`, which means that we’ll need to compute
# the class probabilities ourselves. Notice that we don’t actually care which label is
# associated with each probability, only what the probabilities are:

# In[4]:


from typing import Any
from collections import Counter

def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))


# In[8]:


print(data_entropy(['a', 'a', 'a']))
print(data_entropy([True, False, False, True]))
print(data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75]))


# ## The Entropy of a Partition

# What we’ve done so far is compute the entropy (think “uncertainty”) of a single set of
# labeled data. Now, each stage of a decision tree involves asking a question whose
# answer partitions data into one or (hopefully) more subsets.
# 
# For instance, our “does it
# have more than five legs?” question partitions animals into those that have more than
# five legs (e.g., spiders) and those that don’t (e.g., echidnas).
# 
# Correspondingly, we’d like some notion of the entropy that results from partitioning a
# set of data in a certain way. We want a partition to have low entropy if it splits the
# data into subsets that themselves have low entropy (i.e., are highly certain), and high
# entropy if it contains subsets that (are large and) have high entropy (i.e., are highly
# uncertain).
# 
# For example, the “Australian five-cent coin” question was pretty dumb, as it partitioned the remaining animals at that point into $S_1$ = {echidna} and
# $S_2$ = {everything else}, where $S_2$ is both large and high-entropy. ($S_1$ has no entropy,
# but it represents a small fraction of the remaining “classes.”)
# 
# Mathematically, if we partition our data $S$ into subsets $S_1, \dots , S_m$ containing proportions
# $q_1, \dots, q_m$ of the data, then we compute the entropy of the partition as a weighted
# sum:
# 
# $$ H = q_1 H(S_1) + \dots + q_m H(S_m) $$
# 
# which we can implement as:

# In[9]:


def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)


# Note: One problem with this approach is that partitioning by an attribute
# with many different values will result in a very low entropy due to
# overfitting.
# 
# For example, imagine you work for a bank and are trying
# to build a decision tree to predict which of your customers are
# likely to default on their mortgages, using some historical data as
# your training set. Imagine further that the dataset contains each
# customer’s Social Security number.
# 
# Partitioning on SSN will produce
# one-person subsets, each of which necessarily has zero
# entropy. But a model that relies on SSN is *certain* not to generalize
# beyond the training set. For this reason, you should probably try to
# avoid (or bucket, if appropriate) attributes with large numbers of
# possible values when creating decision trees.
