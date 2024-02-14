#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes (Contineud)

# In[ ]:


from Week06R import *


# ## Using Our Model

# A popular (if somewhat old) dataset is the [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/). We’ll look at
# the files prefixed with 20021010.
# 
# Here is a script that will download and unpack them to the directory of your choice
# (or you can do it manually):

# In[ ]:


from io import BytesIO # So we can treat bytes as a file.
import requests # To download the files, which
import tarfile # are in .tar.bz format.

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]

# This is where the data will end up,
# in /spam, /easy_ham, and /hard_ham subdirectories.
# Change this to where you want the data.
OUTPUT_DIR = 'spam_data'

for filename in FILES:
    # Use requests to get the file contents at each URL.
    content = requests.get(f"{BASE_URL}/{filename}").content
    
    # Wrap the in-memory bytes so we can use them as a "file."
    fin = BytesIO(content)
    
    # And extract all the files to the specified output dir.
    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)


# After downloading the data you should have three folders: spam, easy_ham, and
# hard_ham. Each folder contains many emails, each contained in a single file. To keep
# things really simple, we’ll just look at the subject lines of each email.
# 
# How do we identify the subject line? When we look through the files, they all seem to
# start with “Subject:”. So we’ll look for that:

# In[ ]:


import glob, re

# modify the path to wherever you've put the files
path = 'spam_data/*/*'

data: List[Message] = []

# glob.glob returns every filename that matches the wildcarded path
for filename in glob.glob(path):
    is_spam = "ham" not in filename

    # There are some garbage characters in the emails, the errors='ignore'
    # skips them instead of raising an exception.
    with open(filename, errors='ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break  # done with this file


# Now we can split the data into training data and test data, and then we’re ready to
# build a classifier:

# In[ ]:


# The same split_data function that we've used earlier
import random
from typing import TypeVar, Tuple
X = TypeVar('X')  # generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    random.shuffle(data)              # because shuffle modifies the list.
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.


# In[ ]:


random.seed(0) # For illustrative purposes only, so that you will also get the same answer when you execute the code
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)


# Let’s generate some predictions and check how our model does:

# In[ ]:


from collections import Counter

predictions = [(message, model.predict(message.text))
               for message in test_messages]

print("********************************************************************************")
print(predictions[:5])
print("********************************************************************************")


# Assume that spam_probability > 0.5 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)

print(f"confusion matrix = {confusion_matrix}")


# This gives 86 true positives (spam classified as “spam”), 29 false positives (ham classified
# as “spam”), 670 true negatives (ham classified as “ham”), and 40 false negatives
# (spam classified as “ham”). This means our precision is 86 / (86 + 29) = 75%, and our
# recall is 86 / (86 + 40) = 68%, which are not bad numbers for such a simple model.
# (Presumably we’d do better if we looked at more than the subject lines.)
# 
# We can also inspect the model’s innards to see which words are least and most indicative
# of spam:

# In[ ]:


def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    # We probably shouldn't call private methods, but it's for a good cause.
    prob_if_spam, prob_if_ham = model._probabilities(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])


# The spammiest words include things like *sale*, *mortgage*, *money*, and *rates*, whereas
# the hammiest words include things like *spambayes*, *users*, *apt*, and *perl*. So that also
# gives us some intuitive confidence that our model is basically doing the right thing.
# 
# How could we get better performance? One obvious way would be to get more data
# to train on. There are a number of ways to improve the model as well. Here are some
# possibilities that you might try:
# - Look at the message content, not just the subject line.
# - Our classifier takes into account every word that appears in the training set, even words that appear only once. Modify the classifier to accept an optional `min_count` threshold and ignore tokens that don’t appear at least that many times.
# - The tokenizer has no notion of similar words (e.g., *cheap* and *cheapest*). Modify the classifier to take an optional `stemmer` function that converts words to *equivalence classes* of words. Creating a good stemmer function is hard. People frequently use the [Porter stemmer](https://tartarus.org/martin/PorterStemmer/). But for illustrative purposes, here is a really simple stemmer function:
# ```python
# def drop_final_s(word):
#     return re.sub("s$", "", word)
# ```
# - Although our features are all of the form “message contains word $w_i$,” there’s no reason why this has to be the case. We could add extra features like “message contains a number”.
# 
#     
# 

# ## Naive Bayes from scikit-learn

# The `scikit-learn` package contains a few variations of the Naive Bayes model. See the documentations at:
# 
# https://scikit-learn.org/stable/modules/naive_bayes.html
# 
# In particular, its `BernoulliNB` model is considered to be equivalent to what we have implemented. Each feature is assumed to be a binary-valued (Bernoulli, boolean) variable, or will be converted to binary features if given other types of data.

# In[ ]:


import numpy as np
rng = np.random.RandomState(0)
X = rng.randint(2, size=(10, 10))
Y = rng.randint(2, size = 10)

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
clf.fit(X, Y)
print(f"actual    = {Y}")
print(f"predicted = {clf.predict(X)}")


# Another variant, `GaussianNB`, is capable of classifying the *iris* dataset that we've looked at earlier. Without getting into too much into the statistical and mathematical details, you can think of the `GaussianNB` model as capable of handling input features that are numeric floating point values, while the `BernoulliNB` model is able to handle binary input features (e.g., 0 and 1). Of course, there are also other types of features (e.g., categorical input features) that may require other appropriate Naive Bayes variants.

# In[ ]:


# Adapted from https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# load the iris datasets
dataset = datasets.load_iris()

# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

