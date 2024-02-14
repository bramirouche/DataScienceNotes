#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes

# ## A Really Dumb Spam Filter

# Imagine a “universe” that consists of receiving a message chosen randomly from all
# possible messages. Let $S$ be the event “the message is spam” and $B$ be the event “the
# message contains the word *bitcoin*.” Bayes’s theorem tells us that the probability that
# the message is spam conditional on containing the word *bitcoin* is:
# 
# $$ P(S|B) = \frac{P(B|S)P(S)}{P(B|S)P(S) + P(B|\neg S)P(\neg S)} $$
# 
# The numerator is the probability that a message is spam *and* contains *bitcoin*, while
# the denominator is just the probability that a message contains *bitcoin*.
# 
# Hence, you
# can think of this calculation as simply representing the proportion of *bitcoin* messages
# that are spam.
# 
# If we have a large collection of messages we know are spam, and a large collection of
# messages we know are not spam, then we can easily estimate $P(B|S)$ and $P(B|\neg S)$.
# 
# If we further assume that any message is equally likely to be spam or not spam (so that
# $P(S) = P(\neg S) = 0.5$), then:
# 
# $$ P(S|B) = \frac{P(B|S)}{P(B|S) + P(B|\neg S)} $$
# 
# For example, if 50% of spam messages have the word bitcoin, but only 1% of nonspam
# messages do, then the probability that any given bitcoin-containing email is spam is:
# 
# $$ \frac{0.5}{0.5 + 0.01} \approx 0.98 $$

# ## A More Sophisticated Spam Filter

# Imagine now that we have a vocabulary of many words, $w_1, \dots, w_n$. To move this into
# the realm of probability theory, we’ll write $X_i$ for the event “a message contains the
# word $w_i$.”
# 
# Also imagine that (through some unspecified-at-this-point process) we’ve
# come up with an estimate $P(X_i|S)$ for the probability that a spam message contains the
# $i$th word, and a similar estimate $P(X_i|\neg S)$ for the probability that a nonspam message
# contains the $i$th word.
# 
# The key to Naive Bayes is making the (big) assumption that the presences (or absences)
# of each word are independent of one another, conditional on a message being
# spam or not.
# 
# Intuitively, this assumption means that knowing whether a certain spam
# message contains the word *bitcoin* gives you no information about whether that same
# message contains the word *rolex*.
# 
# In math terms, this means that:
# 
# $$ P(X_1 = x_1, \dots, X_n = x_n | S) = P(X_1 = x_1 | S) \times \dots \times P(X_n = x_n | S) $$
# 
# This is an extreme assumption. (There’s a reason the technique has *naive* in its name.)
# 
# Imagine that our vocabulary consists only of the words *bitcoin* and *rolex*, and that half
# of all spam messages are for “earn bitcoin” and that the other half are for “authentic
# rolex.” In this case, the Naive Bayes estimate that a spam message contains both *bitcoin*
# and *rolex* is:
# 
# $$ P(X_1 = 1, X_2 = 1 | S) = P(X_1 = 1 | S) P(X_2 = 1 | S) = 0.5 \times 0.5 = 0.25 $$
# 
# while in reality that bitcoin and rolex actually never occur together (in our example).
# 
# Despite the unrealisticness of this assumption, this model often performs
# well and has historically been used in actual spam filters.
# 
# The same Bayes’s theorem reasoning we used for our “bitcoin-only” spam filter tells
# us that we can calculate the probability a message is spam using the equation:
# 
# $$ P(S|X=x) = \frac{P(X=x|S)}{P(X=x|S) + P(X=x|\neg S)} $$
# 
# The Naive Bayes assumption allows us to compute each of the probabilities on the
# right simply by multiplying together the individual probability estimates for each
# vocabulary word.
# 
# In practice, you usually want to avoid multiplying lots of probabilities together, to
# prevent a problem called *underflow*, in which computers don’t deal well with floatingpoint
# numbers that are too close to 0.
# 
# Recalling from algebra that $\log(ab) = \log(a) + \log(b)$ and that $e^{\log(x)} = x$, we usually compute $p_1 * \dots * p_n$ as the equivalent (but floating-point-friendlier):
# 
# $$ \Large e^{\log(p_1) + \dots + \log(p_n)} $$
# 
# The only challenge left is coming up with estimates for $P(X_i|S)$ and $P(X_i|\neg S)$, the probabilities that a spam message (or nonspam message) contains the word $w_i$. If we
# have a fair number of “training” messages labeled as spam and not spam, an obvious
# first try is to estimate $P(X_i|S)$ simply as the fraction of spam messages containing the
# word $w_i$.
# 
# This causes a big problem, though. Imagine that in our training set the vocabulary
# word *data* only occurs in nonspam messages. Then we’d estimate $P(data|S) = 0$. The
# result is that our Naive Bayes classifier would always assign spam probability 0 to any
# message containing the word *data*, even a message like “data on free bitcoin and
# authentic rolex watches.” To avoid this problem, we usually use some kind of smoothing.
# 
# In particular, we’ll choose a *pseudocount*—$k$—and estimate the probability of seeing
# the *i*th word in a spam message as:
# 
# $$ P(X_i|S) = \frac{\textrm{number of spams containing } w_i + k}{\textrm{number of spams } + 2k} $$
# 
# Usually $k$ is just some very small positive number, and we've essentially added $k$ to the numerator and $2k$ to the denominator to prevent the fraction to ever become 0.
# 
# We do similarly for $P(X_i | \neg S)$. That is, when computing the spam probabilities for the
# *i*th word, we assume we also saw $k$ additional nonspams containing the word and $k$
# additional nonspams not containing the word.
# 
# For example, if data occurs in 0/98 spam messages, and if $k$ is 1, we estimate $P(data|S)$
# as 1/100 = 0.01, which allows our classifier to still assign some nonzero spam probability
# to messages that contain the word data.

# ## Implementation

# Now we have all the pieces we need to build our classifier. First, let’s create a simple
# function to tokenize messages into distinct words. We’ll first convert each message to
# lowercase, then use `re.findall` to extract “words” consisting of letters, numbers, and
# apostrophes. Finally, we’ll use `set` to get just the distinct words:

# In[1]:


from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # Convert to lowercase,
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)                       # remove duplicates.

print(tokenize("Data Science is science"))


# We’ll also define a type for our training data:

# In[3]:


from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool


# As our classifier needs to keep track of tokens, counts, and labels from the training
# data, we’ll make it a class. Following convention, we refer to nonspam emails as ham
# emails.
# 
# Let's take a look at the `NaiveBayesClassifier` class and we will cover each part one at a time:

# In[4]:


from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | not spam)"""
        spam = self.token_spam_counts[token] # number of spams containing the token word
        ham = self.token_ham_counts[token]   # number of hams containing the token word

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary.
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it;
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # otherwise add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


# The constructor will take just one parameter, the pseudocount to use when computing
# probabilities. It also initializes an empty set of tokens, counters to track how often
# each token is seen in spam messages and ham messages, and counts of how many
# spam and ham messages it was trained on:
# ```python
#     def __init__(self, k: float = 0.5) -> None:
#         self.k = k  # smoothing factor
# 
#         self.tokens: Set[str] = set()
#         self.token_spam_counts: Dict[str, int] = defaultdict(int)
#         self.token_ham_counts: Dict[str, int] = defaultdict(int)
#         self.spam_messages = self.ham_messages = 0
# ```
# 
# Next, we’ll give it a method to train it on a bunch of messages. First, we increment the
# `spam_messages` and `ham_messages` counts. Then we tokenize each message text, and
# for each token we increment the `token_spam_counts` or `token_ham_counts` based on
# the message type:
# 
# ```python
#     def train(self, messages: Iterable[Message]) -> None:
#         for message in messages:
#             # Increment message counts
#             if message.is_spam:
#                 self.spam_messages += 1
#             else:
#                 self.ham_messages += 1
# 
#             # Increment word counts
#             for token in tokenize(message.text):
#                 self.tokens.add(token)
#                 if message.is_spam:
#                     self.token_spam_counts[token] += 1
#                 else:
#                     self.token_ham_counts[token] += 1
# ```
# 
# Ultimately we’ll want to predict $P(\textrm{spam }|\textrm{ token})$. As we saw earlier, to apply Bayes’s
# theorem we need to know $P(\textrm{token }|\textrm{ spam})$ and $P(\textrm{token }|\textrm{ ham})$ for each token in the
# vocabulary. So we’ll create a “private” helper function to compute those:
# 
# ```python
#     def _probabilities(self, token: str) -> Tuple[float, float]:
#         """returns P(token | spam) and P(token | not spam)"""
#         spam = self.token_spam_counts[token] # number of spams containing the token word
#         ham = self.token_ham_counts[token]   # number of hams containing the token word
# 
#         p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
#         p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
# 
#         return p_token_spam, p_token_ham
# ```
# 
# Finally, we’re ready to write our `predict` method. As mentioned earlier, rather than
# multiplying together lots of small probabilities, we’ll instead sum up the log probabilities:
# 
# ```python
#     def predict(self, text: str) -> float:
#         text_tokens = tokenize(text)
#         log_prob_if_spam = log_prob_if_ham = 0.0
# 
#         # Iterate through each word in our vocabulary.
#         for token in self.tokens:
#             prob_if_spam, prob_if_ham = self._probabilities(token)
# 
#             # If *token* appears in the message,
#             # add the log probability of seeing it;
#             if token in text_tokens:
#                 log_prob_if_spam += math.log(prob_if_spam)
#                 log_prob_if_ham += math.log(prob_if_ham)
# 
#             # otherwise add the log probability of _not_ seeing it
#             # which is log(1 - probability of seeing it)
#             else:
#                 log_prob_if_spam += math.log(1.0 - prob_if_spam)
#                 log_prob_if_ham += math.log(1.0 - prob_if_ham)
# 
#         prob_if_spam = math.exp(log_prob_if_spam)
#         prob_if_ham = math.exp(log_prob_if_ham)
#         return prob_if_spam / (prob_if_spam + prob_if_ham)
# ```

# # Testing Our Model

# Let’s make sure our model works by writing some unit tests for it.

# In[5]:


messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)


# First, let’s check that it got the counts right:

# In[6]:


print(model.tokens)            # {"spam", "ham", "rules", "hello"}
print(model.spam_messages)     # 1
print(model.ham_messages)      # 2
print(model.token_spam_counts) # {"spam": 1, "rules": 1}
print(model.token_ham_counts)  # {"ham": 2, "rules": 1, "hello": 1}


# Now let’s make a prediction. We’ll also (laboriously) go through our Naive Bayes logic
# by hand, and make sure that we get the same result:

# In[7]:


text = "hello spam"

# P(token | spam) = (num of spams in training data containing token + k) / (total number of spams in training data + 2k)
probs_if_spam = [
        (1 + 0.5) / (1 + 2 * 0.5),  # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
        (0 + 0.5) / (1 + 2 * 0.5)   # "hello" (present)
]

# P(token | ham) = (num of hams in training data containing token + k) / (total number of hams in training data + 2k)
probs_if_ham = [
        (0 + 0.5) / (2 + 2 * 0.5),  # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
        (1 + 0.5) / (2 + 2 * 0.5),  # "hello" (present)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Both should be about 0.83
print(model.predict(text))
print(p_if_spam / (p_if_spam + p_if_ham))


# This test passes, so it seems like our model is doing what we think it is. If you look at
# the actual probabilities, the two big drivers are that our message contains *spam* (which our lone training spam message did) and that it doesn’t contain ham (which
# both our training ham messages did).
# 
# Next we will try it on some real data.
