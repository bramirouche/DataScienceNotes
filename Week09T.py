#!/usr/bin/env python
# coding: utf-8

# # Decision Trees (Continued)

# In[1]:


from Week09M import *


# ## Creating a Decision Tree

# The VP provides you with the interviewee data, consisting of (per your specification)
# a `NamedTuple` of the relevant attributes for each candidate—level, preferred
# language, active on Twitter or not, has a PhD or not, and whether the candidate interviewed well:

# In[2]:


from typing import NamedTuple, Optional

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data


# In[3]:


#  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
         ]


# Our tree will consist of *decision nodes* (which ask a question and direct us differently
# depending on the answer) and leaf nodes (which give us a prediction).
# 
# We will build it
# using the relatively simple ID3 algorithm, which operates in the following manner.
# Let’s say we’re given some labeled data, and a list of attributes to consider branching
# on:
# 
# - If the data all have the same label, create a leaf node that predicts that label and then stop.
# - If the list of attributes is empty (i.e., there are no more possible questions to ask), create a leaf node that predicts the most common label and then stop.
# - Otherwise, try partitioning the data by each of the attributes.
# - Choose the partition with the lowest partition entropy.
# - Add a decision node based on the chosen attribute.
# - Recur on each partitioned subset using the remaining attributes.
# 
# This is what’s known as a “greedy” algorithm because, at each step, it chooses the
# most immediately best option. Given a dataset, there may be a better tree with a
# worse-looking first move. If so, this algorithm won’t find it. Nonetheless, it is relatively
# easy to understand and implement, which makes it a good place to begin
# exploring decision trees.
# 
# Let’s manually go through these steps on the interviewee dataset. The dataset has both
# `True` and `False` labels, and we have four attributes we can split on. So our first step
# will be to find the partition with the least entropy. We’ll start by writing a function
# that does the partitioning:

# In[4]:


from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar('T')  # generic type for inputs/data point, for our purposes, T represents a candidate

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute."""
    partitions: Dict[Any, List[T]] = defaultdict(list) # the keys are the possible values for a given attribute to split on
    for input in inputs:
        # getattr() returns value of the named attribute of the given object, you can think of it as Python's built-in getter 
        # for any attribute defined within a class
        # e.g., Given the first candidate object as 'input' and attribute is 'level', getattr() would return 'Senior'
        key = getattr(input, attribute)  # value of the specified attribute
        partitions[key].append(input)    # add input to the correct partition
    return partitions


# and one that uses it to compute entropy:

# In[5]:


def partition_entropy_by(inputs: List[T],
                         attribute: str,
                         label_attribute: str) -> float:
    """Compute the entropy corresponding to the given partition"""
    # partitions consist of our inputs
    partitions = partition_by(inputs, attribute)

    # but partition_entropy needs just the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)


# Then we just need to find the minimum-entropy partition for the whole dataset:

# In[6]:


for key in ['level','lang','tweets','phd']:
    print(key, partition_entropy_by(inputs, key, 'did_well'))


# The lowest entropy comes from splitting on `level`, so we’ll need to make a subtree
# for each possible level value.
# 
# Every `Mid` candidate is labeled True, which means that
# the `Mid` subtree is simply a leaf node predicting `True`. For Senior candidates, we have
# a mix of `True`s and `False`s, so we need to split again:

# In[7]:


senior_inputs = [input for input in inputs if input.level == 'Senior']

for key in ['lang','tweets','phd']:
    print(key, partition_entropy_by(senior_inputs, key, 'did_well'))


# This shows us that our next split should be on `tweets`, which results in a zeroentropy
# partition. For these `Senior`-level candidates, “yes” tweets always result in
# `True` while “no” tweets always result in `False`.
# 
# We will also do the same thing for the Junior candidates:

# In[8]:


junior_inputs = [input for input in inputs if input.level == 'Junior']

for key in ['lang','tweets','phd']:
    print(key, partition_entropy_by(junior_inputs, key, 'did_well'))


# This shows that we should split on `phd`, after which we find that no PhD always results in `True` and PhD always results in
# `False`.
# 
# The complete decision tree would then look like this:
# 
# ![The decision tree for hiring](hiring_tree.jpg)

# ## Putting It All Together

# Now that we’ve seen how the algorithm works, we would like to implement it more
# generally. This means we need to decide how we want to represent trees. We’ll use
# pretty much use the most lightweight representation possible. We define a tree to be
# either:
# - a `Leaf` (that predicts a single value), or
# - a `Split` (containing an attribute to split on, subtrees for specific values of that attribute, and possibly a default value to use if we see an unknown value).

# In[9]:


from typing import NamedTuple, Union, Any

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]


# With this representation, our hiring tree would look like:

# In[10]:


hiring_tree = Split('level', {   # First, consider "level".
    'Junior': Split('phd', {     # if level is "Junior", next look at "phd"
        False: Leaf(True),       #   if "phd" is False, predict True
        True: Leaf(False)        #   if "phd" is True, predict False
    }),
    'Mid': Leaf(True),           # if level is "Mid", just predict True
    'Senior': Split('tweets', {  # if level is "Senior", look at "tweets"
        False: Leaf(False),      #   if "tweets" is False, predict False
        True: Leaf(True)         #   if "tweets" is True, predict True
    })
})


# There’s still the question of what to do if we encounter an unexpected (or missing)
# attribute value. What should our hiring tree do if it encounters a candidate whose
# `level` is `Intern`? We’ll handle this case by populating the `default_value` attribute
# with the most common label.
# 
# Given such a representation, we can classify an input with:

# In[12]:


def classify(tree: DecisionTree, input: Any) -> Any:
    """classify the input using the given decision tree"""

    # If this is a leaf node, return its value
    if isinstance(tree, Leaf):
        return tree.value

    # Otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values of are subtrees to consider next
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:   # If no subtree for key,
        return tree.default_value          # return the default value.

    subtree = tree.subtrees[subtree_key]   # Choose the appropriate subtree
    return classify(subtree, input)        # and use it to classify the input.


# All that’s left is to build the tree representation from our training data:

# In[16]:


DEBUG = False # used for printing out optional outputs below


# In[14]:


def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    # Count target labels
    label_counts = Counter(getattr(input, target_attribute)
                           for input in inputs)
    if (DEBUG): print(f"label_counts = {label_counts}")
    most_common_label = label_counts.most_common(1)[0][0]
    if (DEBUG): print(f"label_counts.most_common(1) = {label_counts.most_common(1)}")

    # If there's a unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # If no split attributes left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)

    # Otherwise split by the best attribute

    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute"""
        return partition_entropy_by(inputs, attribute, target_attribute)

    # for each attribute in split_attributes, get the partition's entropy
    # then sort the all the attributes by the entropy
    best_attribute = min(split_attributes, key=split_entropy)

    # get the actual partition by the best_attribute
    partitions = partition_by(inputs, best_attribute)
    
    # remove the best_attribute from the current list of split_attributes
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # recursively build the subtrees
    subtrees = {attribute_value : build_tree_id3(subset,
                                                 new_attributes,
                                                 target_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)


# In the tree we built, every leaf consisted entirely of True inputs or entirely of False
# inputs. This means that the tree predicts perfectly on the training dataset. But we can
# also apply it to new data that wasn’t in the training set:

# In[17]:


tree = build_tree_id3(inputs,
                      ['level', 'lang', 'tweets', 'phd'],
                      'did_well')


# In[18]:


# Should predict True
print(classify(tree, Candidate("Junior", "Java", True, False)))

# Should predict False
print(classify(tree, Candidate("Junior", "Java", True, True)))


# And also to data with unexpected values:

# In[19]:


# Should predict True because "Intern" does not exists in the tree and the most common (default) value is True
print(classify(tree, Candidate("Intern", "Java", True, True)))


# ## Random Forests

# Given how closely decision trees can fit themselves to their training data, it’s not surprising
# that they have a tendency to overfit. One way of avoiding this is a technique
# called *random forests*, in which we build multiple decision trees and combine their outputs.
# - if they’re classification trees, we might let them vote;
# - if they’re regression trees, we might average their predictions.
# 
# Our tree-building process was deterministic, so how do we get random trees?
# 
# One piece involves *bootstrapping* data (discussed earlier in Regression). Rather than training each tree on all the inputs in the training set, we train
# each tree on the result of `bootstrap_sample(inputs)`. Since each tree is built using
# different data, each tree will be different from every other tree.
# 
# A second source of randomness involves changing the way we choose the
# `best_attribute` to split on. Rather than looking at all the remaining attributes, we
# first choose a random subset of them and then split on whichever of those is best:

# ```python
# # if there are already few enough split candidates, look at all of them
# if len(split_candidates) <= self.num_split_candidates:
#     sampled_split_candidates = split_candidates
# # otherwise pick a random sample
# else:
#     sampled_split_candidates = random.sample(split_candidates,
#                                              self.num_split_candidates)
#     
# # now choose the best attribute only from those candidates
# best_attribute = min(sampled_split_candidates, key=split_entropy)
# partitions = partition_by(inputs, best_attribute)
# ```

# This is an example of a broader technique called *ensemble learning* in which we combine
# several weak learners (typically high-bias, low-variance models) in order to produce
# an overall strong model.
