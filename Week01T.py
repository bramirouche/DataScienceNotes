#!/usr/bin/env python
# coding: utf-8

# # A Crash Course in Python (Continued)

# ## Lists

# Lists are probably the most fundamental data structure in Python. Similar to *arrays* in other languages, but with some added functionality.

# In[1]:


integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, heterogeneous_list, []]

list_length = len(integer_list)     # equals 3
list_sum    = sum(integer_list)     # equals 6

print(list_length)
print(list_sum)


# You can get or set the nth element of a list with square brackets:

# In[2]:


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

zero = x[0]          # equals 0, lists are 0-indexed
one = x[1]           # equals 1
nine = x[-1]         # equals 9, 'Pythonic' for last element
eight = x[-2]        # equals 8, 'Pythonic' for next-to-last element
x[0] = -1            # now x is [-1, 1, 2, 3, ..., 9]

print(f"zero  = {zero}")
print(f"one   = {one}")
print(f"nine  = {nine}")
print(f"eight = {eight}")
print(f"x = {x}")


# You can also use square brackets to slice lists. The slice i:j means all elements from i (**inclusive**) to j (**exclusive**).
# 
# If you leave off the start of the slice, you’ll slice from the beginning of the list, and if you leave of the end of the slice, you’ll slice until the end of the list:

# In[3]:


first_three = x[:3]                 # [-1, 1, 2]
three_to_end = x[3:]                # [3, 4, ..., 9]
one_to_four = x[1:5]                # [1, 2, 3, 4]

# -3 can be interpreted as len(x)-3
last_three = x[-3:]                 # [7, 8, 9]
without_first_and_last = x[1:-1]    # [1, 2, ..., 8]
copy_of_x = x[:]                    # [-1, 1, 2, ..., 9]

print(f"first_three = {first_three}")
print(f"three_to_end = {three_to_end}")
print(f"one_to_four = {one_to_four}")
print(f"last_three = {last_three}")
print(f"without_first_and_last = {without_first_and_last}")
print(f"copy_of_x = {copy_of_x}")


# A slice can take a third argument to indicate its *stride*, which can be negative:

# In[4]:


every_third = x[::3]                 # [-1, 3, 6, 9]
five_to_three = x[5:2:-1]            # [5, 4, 3]

print(f"every_third = {every_third}")
print(f"five_to_three = {five_to_three}")


# Python has an `in` operator to check for list membership:
# 
# Note that this is a linear search algorithm that has the runtime of *O(n)*, where *n* is the length of the list.

# In[5]:


print(1 in [1, 2, 3])    # True
print(0 in [1, 2, 3])    # False


# You can modify a list in place, or you can use `extend` to add items from another collection:

# In[12]:


x = [1, 2, 3]
x.extend([4, 5, 6])     # x is now [1, 2, 3, 4, 5, 6]

print(f"x = {x}")


# If you don’t want to modify `x`, you can use list addition:

# In[7]:


x = [1, 2, 3]
y = x + [4, 5, 6]       # y is [1, 2, 3, 4, 5, 6]; x is unchanged

print(f"x = {x}")
print(f"y = {y}")


# More frequently we will append to lists one item at a time:

# In[8]:


x = [1, 2, 3]
x.append(0)      # x is now [1, 2, 3, 0]
y = x[-1]        # equals 0
z = len(x)       # equals 4

print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")


# It’s often convenient to *unpack* lists when you know how many elements they contain:

# In[15]:


# Must have the same number of values on both sides
x, y = [1, 2]    # now x is 1, y is 2

print(f"x = {x}")
print(f"y = {y}")


# If you do not care about a value, an underscore can be used to represent a value that you’re going to throw away:

# In[16]:


_, y = [1, 2]    # now y == 2, didn't care about the first element

print(f"y = {y}")


# ## Tuples

# Tuples are lists’ **immutable** cousins. Pretty much anything you can do to a list that doesn’t involve modifying it, you can do to a tuple.
# 
# You specify a tuple by using parentheses (or nothing) instead of square brackets:

# In[17]:


my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3      # my_list is now [1, 3]

print(f"my_list = {my_list}")
print(f"my_tuple = {my_tuple}")
print(f"other_tuple = {other_tuple}")

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")


# Tuples are a convenient way to return multiple values from functions:

# In[18]:


def sum_and_product(x, y):
    return (x + y), (x * y)

sp = sum_and_product(2, 3)     # sp is (5, 6)
s, p = sum_and_product(2, 3)   # s is 5, p is 6

print(f"sp = {sp}")
print(f"s  = {s}")
print(f"p  = {p}")


# Tuples (and lists) can also be used for *multiple assignment*:

# In[19]:


x, y = 1, 2     # now x is 1, y is 2

print("before swap:")
print(f"x = {x}, y = {y}")
x, y = y, x     # Pythonic way to swap variables; now x is 2, y is 1
print("after swap:")
print(f"x = {x}, y = {y}")


# ## Dictionaries

# Another fundamental data structure is a dictionary, which associates *values* with *keys* and allows you to quickly retrieve the value corresponding to a given key.
# 
# They are similar to *maps* in other languages. 

# In[20]:


empty_dict = {}                     # Pythonic
empty_dict2 = dict()                # less Pythonic
grades = {"Joel": 80, "Tim": 95}    # dictionary literal

print(f"grades = {grades}")


# You can look up the value for a key using square brackets:

# In[21]:


joels_grade = grades["Joel"]        # equals 80

print(f"joels_grade = {joels_grade}")


# But you’ll get a `KeyError` if you ask for a key that’s not in the dictionary:

# In[22]:


try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate!")


# You can check for the existence of a key using `in`:

# In[23]:


print("Joel" in grades)     # True
print("Kate" in grades)     # False


# Dictionaries have a `get` method that returns a default value (instead of raising an exception) when you look up a key that’s not in the dictionary:

# In[24]:


joels_grade = grades.get("Joel", 0)   # equals 80
kates_grade = grades.get("Kate", 0)   # equals 0
no_ones_grade = grades.get("No One")  # default default is None

print(f"joels_grade = {joels_grade}")
print(f"kates_grade = {kates_grade}")
print(f"no_ones_grade = {no_ones_grade}")


# You can assign (or add) key/value pairs using the same square brackets:

# In[26]:


grades["Tim"] = 99                    # replaces the old value
grades["Kate"] = 100                  # adds a third entry
num_students = len(grades)            # equals 3

print(grades["Tim"])
print(grades["Kate"])
print(num_students)
print(grades)


# Dictionaries can be used to represent structured data:

# In[27]:


tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}

print(tweet)


# Besides looking for specific keys, we can look at all of them:

# In[28]:


tweet_keys   = tweet.keys()     # iterable for the keys
tweet_values = tweet.values()   # iterable for the values
tweet_items  = tweet.items()    # iterable for the (key, value) tuples

print(f"tweet_keys = {tweet_keys}")
print(f"tweet_values = {tweet_values}")
print(f"tweet_items = {tweet_items}")

print("user" in tweet_keys)         # True, but not Pythonic
print("user" in tweet)              # Pythonic way of checking for keys, discussed earlier
print("joelgrus" in tweet_values)   # True (slow but the only way to check)


# ### defaultdict

# A `defaultdict` is like a regular dictionary, except that when you try to look up a key it doesn’t contain, it first adds a value for it using a zero-argument function you provided when you created it.
# 
# In order to use `defaultdict`, you have to import them from `collections`:

# In[30]:


from collections import defaultdict

document = ["data", "science", "from", "scratch", "more", "data"]

word_counts = defaultdict(int)          # int() is a function that produces 0
for word in document:
    word_counts[word] += 1
    
print(word_counts)


# They can also be useful with `list` or `dict`

# In[34]:


dd_list = defaultdict(list)             # list() produces an empty list
dd_list[2].append(1)                    # now dd_list contains {2: [1]}
#dd_list[2] = [1]

dd_dict = defaultdict(dict)             # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle"     # {"Joel" : {"City": Seattle"}}

print(dd_list)
print(dd_dict)


# ## Counters

# A `Counter` turns a sequence of values into a defaultdict(int)-like object mapping keys to counts:

# In[35]:


from collections import Counter
c = Counter(["a", "b", "c", "a"])          # c is (basically) {'a': 2, 'b': 1, 'c': 1}

print(c)


# This gives us a very simple way to solve our word_counts problem:

# In[36]:


# recall, document is a list of words
word_counts = Counter(document)
print(word_counts)


# A `Counter` instance has a `most_common` method that is frequently useful:

# In[41]:


# print the 10 most common words and their counts
for word, count in word_counts.most_common(10):
    print(word, count)


# ## Sets

# Another useful data structure is set, which represents a collection of *distinct* elements.
# 
# You can define a set by listing its elements between curly braces:

# In[ ]:


primes_below_10 = {2, 3, 5, 7}


# However, that doesn’t work for empty sets, as {} already means “empty dict.”
# 
# In that case you’ll need to use `set()` itself:

# In[42]:


s = set()
s.add(1)       # s is now {1}
s.add(2)       # s is now {1, 2}
s.add(2)       # s is still {1, 2}
x = len(s)     # equals 2
y = 2 in s     # equals True
z = 3 in s     # equals False

print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")


# The `in` operation is very fast on sets, making it more appropriate for membership tests than a list:

# In[43]:


hundreds_of_other_words = ["fill", "in", "tons", "of", "words", "in", "here"]  # required for the below code to run

stopwords_list = ["a", "an", "at"] + hundreds_of_other_words + ["yet", "you"]

print("zip" in stopwords_list)     # False, but have to check every element

stopwords_set = set(stopwords_list)
print("zip" in stopwords_set)      # very fast to check


# It's also very convienient to find the *distinct* items in a collection using `set`.

# In[44]:


item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)                # 6
item_set = set(item_list)                 # {1, 2, 3}
num_distinct_items = len(item_set)        # 3
distinct_item_list = list(item_set)       # [1, 2, 3]

print(f"num_items = {num_items}")
print(f"num_distinct_items = {num_distinct_items}")
print(f"distinct_item_list = {distinct_item_list}")

