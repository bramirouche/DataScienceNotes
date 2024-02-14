#!/usr/bin/env python
# coding: utf-8

# # KNN from scikit-learn

# Documentation for KNN from scikit-learn avaialable at https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# In[ ]:


# Adapted from https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
# k-Nearest Neighbor
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# load iris the datasets
dataset = datasets.load_iris()
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()

# data
print(f"type(data) = {type(dataset.data)}")
print(f"data = {dataset.data}")
print(f"type(target) = {type(dataset.target)}")
print(f"target = {dataset.target}")

# preprocessing may be needed on target to convert strings to ints
model.fit(dataset.data, dataset.target)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
# support: the number of occurrences of each class
# macro average: averaging the unweighted mean per label
# weighted average: averaging the support-weighted mean per label
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# # Probability and Bayes' Theorem

# The next model we are interested in is the Naive Bayes classifier. However, it would be quite difficult to understand how it works without some basic understanding of probablility and Bayes' Theorem. So we will briefly introduce these topics.
# 
# For our purposes you should think of probability as a way of quantifying the uncertainty
# associated with *events* chosen from some *universe* of events. Rather than getting
# technical about what these terms mean, think of rolling a die.
# 
# The universe
# consists of all possible outcomes. And any subset of these outcomes is an event; for
# example, “the die rolls a 1” or “the die rolls an even number.”
# 
# Notationally, we write $P(E)$ to mean “the probability of the event $E$.”

# ## Dependence and Independence

# Roughly speaking, we say that two events $E$ and $F$ are *dependent* if knowing something
# about whether $E$ happens gives us information about whether $F$ happens (and
# vice versa). Otherwise, they are *independent*.
# 
# For instance, if we flip a fair coin twice, knowing whether the first flip is heads gives
# us no information about whether the second flip is heads. These events are independent.
# 
# On the other hand, knowing whether the first flip is heads certainly gives us information about whether both flips are tails. (If the first flip is heads, then definitely
# it’s not the case that both flips are tails.) These two events are dependent.
# 
# Mathematically, we say that two events E and F are independent if the probability that
# they both happen is the product of the probabilities that each one happens:
# 
# $$P(E, F) = P(E)P(F)$$
# 
# | 1st Flip | 2nd Flip |
# |:--------:|:--------:|
# | H        | H        |
# | H        | T        |
# | T        | H        |
# | T        | T        |
# 
# In the example, the probability of “first flip heads” is $\frac{1}{2}$, and the probability of “both
# flips tails” is $\frac{1}{4}$, but the probability of “first flip heads *and* both flips tails” is 0. Since $0 \ne \frac{1}{2} * \frac{1}{4}$, the two events are said to be dependent (or equivalently, not independent).
# 
# On the other hand, the probability of "second flip heads" is 1/2, and the probablility of "both flips heads" is 1/4, and since $\frac{1}{4} = \frac{1}{2} * \frac{1}{2}$, the events "first flip heads" and "second flip heads" are said to be independent.

# ## Conditional Probability

# When two events $E$ and $F$ are independent, then by definition we have:
# 
# $$P(E, F) = P(E)P(F)$$
# 
# If they are not necessarily independent (and if the probability of $F$ is not zero), then
# we define the probability of $E$ “conditional on $F$” as:
# 
# $$P(E|F) = \frac{P(E, F)}{P(F)} $$
# 
# You should think of this as the probability that $E$ happens, given that we know that $F$
# happens.
# 
# The $P(F)$ in the denominator can be thought of as first filtering out all the cases where $F$ happens, and among all such cases, look for the cases that $E$ also happens (the numerator).
# 
# For example, let $E$ = "both flips tail" and $F$  = "first flip tail", and we are looking for $P(E|F)$.
# 
# First by looking at the table above, there are 2 cases (bottom 2 rows) where $F$ happens, among which, there is 1 case the $E$ also happens (bottom row), so $P(E|F)$ should be $\frac{1}{2}$.
# 
# Now mathematically if we apply the formula, $P(E, F) = \frac{1}{4}$ and $P(F) = \frac{1}{2}$, so
# 
# $$ P(E|F) = \frac{\frac{1}{4}}{\frac{1}{2}} = \frac{1}{2}$$

# The conditional probablity definition is often rewritten as:
# 
# $$P(E, F) = P(E|F)P(F) $$
# 
# When $E$ and $F$ are independent, you can check that this gives:
# 
# $$P(E|F) = P(E)$$
# 
# which is the mathematical way of expressing that knowing $F$ occurred gives us no
# additional information about whether $E$ occurred.
# 
# One common tricky example involves a family with two (unknown) children. If we
# assume that:
# - Each child is equally likely to be a boy or a girl.
# - The gender of the second child is independent of the gender of the first child.
# 
# | 1st Child | 2nd Child |
# |:--------:|:--------:|
# | Boy        | Boy        |
# | Boy        | Girl        |
# | Girl        | Boy        |
# | Girl        | Girl        |
# 
# Then the event “no girls” has probability 1/4, the event “one girl, one boy” has probability
# 1/2, and the event “two girls” has probability 1/4.
# 
# Now we can ask what is the probability of the event “both children are girls” ($B$) conditional
# on the event “the older child is a girl” ($G$)? Using the definition of conditional
# probability:
# 
# $$ P(B|G) = \frac{P(B, G)}{P(G)} = \frac{P(B)}{P(G)} = \frac{\frac{1}{4}}{\frac{1}{2}} = 1/2 $$
# 
# since the event $B$ and $G$ (“both children are girls *and* the older child is a girl”) is just
# the event $B$. (Once you know that both children are girls, it’s necessarily true that the
# older child is a girl.)
# 
# Most likely this result accords with your intuition.
# 
# We could also ask about the probability of the event “both children are girls” conditional
# on the event “at least one of the children is a girl” ($L$). Surprisingly, the answer
# is different from before!
# 
# As before, the event $B$ and $L$ (“both children are girls *and* at least one of the children
# is a girl”) is just the event $B$. This means we have:
# 
# $$ P(B|L) = \frac{P(B, L)}{P(L)} = \frac{P(B)}{P(L)} = \frac{\frac{1}{4}}{\frac{3}{4}} =1/3 $$
# 
# How can this be the case? Well, if all you know is that at least one of the children is a
# girl, then it is twice as likely that the family has one boy and one girl than that it has
# both girls.
# 
# We can check this by “generating” a lot of families:

# In[ ]:


import enum, random

# An Enum is a typed set of enumerated values. We can use them
# to make our code more descriptive and readable.
class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

# random.seed(0)

for _ in range(10000):
    older = random_kid()
    younger = random_kid()
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print("P(both | older):", both_girls / older_girl)     #  ~ 1/2
print("P(both | either): ", both_girls / either_girl)  #  ~ 1/3


# ## Bayes’ Theorem

# One of the data scientist’s best friends is Bayes’s theorem, which is a way of “reversing”
# conditional probabilities.
# 
# Let’s say we need to know the probability of some event
# $E$ conditional on some other event $F$ occurring. But we only have information about
# the probability of $F$ conditional on $E$ occurring.
# 
# Using the definition of conditional
# probability twice tells us that:
# 
# $$ P(E|F) = \frac{P(E,F)}{P(F)} = \frac{P(F|E)P(E)}{P(F)} $$
# 
# The event $F$ can be split into the two mutually exclusive events “$F$ and $E$” and “$F$ and
# not $E$.” If we write $\neg E$ for “not $E$” (i.e., “$E$ doesn’t happen”), then:
# 
# $$ P(F) = P(F, E) + P(F, \neg E) $$
# 
# so that:
# 
# $$ P(E|F) = \frac{P(F|E)P(E)}{P(F|E)P(E) + P(F|\neg E)P(\neg E)} $$
# 
# which is how Bayes’s theorem is often stated.

# This theorem often gets used to demonstrate why data scientists are smarter than
# doctors.
# 
# Imagine a certain disease that affects 1 in every 10,000 people. And imagine
# that there is a test for this disease that gives the correct result (“diseased” if you have
# the disease, “nondiseased” if you don’t) 99% of the time.
# 
# What does a positive test mean? Let’s use $T$ for the event “your test is positive” and $D$
# for the event “you have the disease.” Then Bayes’s theorem says that the probability
# that you have the disease, conditional on testing positive, is:
# 
# $$ P(D|T) = \frac{P(T|D)P(D)}{P(T|D)P(D) + P(T|\neg D)P(\neg D)} $$
# 
# Here we know that $P(T|D)$ , the probability that someone with the disease tests positive,
# is 0.99.
# 
# $P(D)$, the probability that any given person has the disease, is 1/10,000 = 0.0001.
# 
# $P(T|\neg D)$ , the probability that someone without the disease tests positive, is 0.01.
# 
# And $P(\neg D)$ , the probability that any given person doesn’t have the disease, is
# 0.9999.
# 
# If you substitute these numbers into Bayes’s theorem, you find:
# 
# $$ P(D|T) = \frac{0.99 * 0.0001}{0.99 * 0.0001 + 0.01 * 0.9999} \approx 0.0098 $$
# 
# That is, less than 1% of the people who test positive actually have the disease.
# 
# A more intuitive way to see this is to imagine a population of 1 million people. You’d
# expect 100 of them to have the disease, and 99 of those 100 to test positive. On the
# other hand, you’d expect 999,900 of them not to have the disease, and 9,999 of those
# to test positive. That means you’d expect only 99 out of (99 + 9999) positive testers to
# actually have the disease.
# 
# Note: This assumes that people take the test more or less at random. If
# only people with certain symptoms take the test, we would instead
# have to condition on the event “positive test *and* symptoms” and
# the number would likely be a lot higher.

# ## Random Variables

# A *random variable* is a variable whose possible values have an associated probability
# distribution.
# 
# A very simple random variable equals 1 if a coin flip turns up heads and
# 0 if the flip turns up tails.
# 
# A more complicated one might measure the number of
# heads you observe when flipping a coin 10 times or a value picked from `range(10)`
# where each number is equally likely.
# 
# The associated distribution gives the probabilities that the variable realizes each of its
# possible values.
# 
# The coin flip variable equals 0 with probability 0.5 and 1 with probability
# 0.5.
# 
# The `range(10)` variable has a distribution that assigns probability 0.1 to
# each of the numbers from 0 to 9.
# 
# We will sometimes talk about the expected value of a random variable, which is the
# average of its values weighted by their probabilities.
# 
# The coin flip variable has an
# expected value of 1/2 (= 0 * 1/2 + 1 * 1/2), and the `range(10)` variable has an
# expected value of 4.5.
# 
# Random variables can be conditioned on events just as other events can.
# 
# Going back
# to the two-child example from the "Conditional Probability" section, if $X$ is the random variable representing the number of girls, $X$ equals 0 with probability 1/4, 1 with
# probability 1/2, and 2 with probability 1/4.
# 
# We can define a new random variable $Y$ that gives the number of girls conditional on
# at least one of the children being a girl. Then $Y$ equals 1 with probability 2/3 and 2 with probability 1/3. And a variable $Z$ that’s the number of girls conditional on the
# older child being a girl equals 1 with probability 1/2 and 2 with probability 1/2.
# 
# For the most part, we will be using random variables *implicitly* in what we do without
# calling special attention to them. But if you look deeply you’ll see them.
