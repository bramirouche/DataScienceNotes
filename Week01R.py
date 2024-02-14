#!/usr/bin/env python
# coding: utf-8

# # A Crash Course in Python (Continued)

# ## Control Flow

# As in most programming languages, you can perform an action conditionally using `if`:

# In[ ]:


if 1 > 2:
    message = "if only 1 were greater than two..."
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"
    
print(message)


# You can also write a *ternary* if-then-else on one line, which we will do occasionally:

# In[ ]:


x = 1
parity = "even" if x % 2 == 0 else "odd"
print(parity)


# Python has a `while` loop:

# In[ ]:


x = 0
while x < 10:
    print(f"{x} is less than 10")
    x += 1


# although more often we’ll use `for` and `in`:

# In[ ]:


# range(10) is the numbers 0, 1, ..., 9
for x in range(10):
    print(f"{x} is less than 10")


# If you need more complex logic, you can use `continue` and `break`:

# In[ ]:


for x in range(10):
    if x == 3:
        continue  # go immediately to the next iteration
    if x == 5:
        break     # quit the loop entirely
    print(x)


# ## Truthiness

# Booleans in Python work as in most other languages, except that they’re capitalized:

# In[ ]:


one_is_less_than_two = 1 < 2          # equals True
true_equals_false = True == False     # equals False

print(one_is_less_than_two)
print(true_equals_false)


# Python uses the value `None` to indicate a nonexistent value. It is similar to other languages’ `null`:

# In[ ]:


x = None

# assert will throw an exception if the condition is False, discussed in details later
assert x == None, "this is the not the Pythonic way to check for None"
assert x is None, "this is the Pythonic way to check for None"


# There is an subtle but important difference between `==` and `is`
# * `==` checks if the contents/values of the two objects are the same
# * `is` checks if the identities/memory address of the two objects are the same

# In[ ]:


list1 = [1, 2]
list1a = list1
print(list1a == list1)
print(list1a is list1)

list2 = [1, 2]
print(list2 == list1)
print(list2 is list1)


# Python lets you use any value where it expects a Boolean. The following are all “falsy”:
# * `False`
# * `None`
# * `[]` (an empty `list`)
# * `{}` (an empty `dict`)
# * `""`
# * `set()`
# * `0`
# * `0.0`
# 
# Pretty much anything else gets treated as `True`. This allows you to easily use `if` statements to test for empty lists, empty strings, empty dictionaries, and so on.

# In[ ]:


def some_function_that_returns_a_string():
    return ""

s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""
    
print(len(first_char))


# A shorter (but possibly more confusing) way of doing the same is:

# In[ ]:


first_char = s and s[0]
print(len(first_char))


# since `and` returns its second value when the first is “truthy,” and the first value when it’s not (short circuit).
# 
# Similarly, if x is either a number or possibly None:

# In[ ]:


x = 1 # None # try a number, then None

safe_x = x or 0

print(safe_x)


# `safe_x` is definitely a number, although:

# In[ ]:


safe_x = x if x is not None else 0

print(safe_x)


# is possibly more readable.

# Python has an `all` function, which takes an iterable and returns `True` precisely when every element is truthy (or when the iterable contains no element), and an `any` function, which returns `True` when at least one element is truthy.
# 
# Note: an iterable is any Python object capable of returning its members one at a time, permitting it to be iterated over in a for-loop, e.g., lists, tuples, strings, etc.

# In[ ]:


print(all([True, 1, {3}]))   # True, all are truthy
print(all([True, 1, {}]))    # False, {} is falsy
print(any([True, 1, {}]))    # True, True is truthy
print(all([]))               # True, no falsy elements in the list
print(any([]))               # False, no truthy elements in the list


# ## Sorting

# Every Python list has a `sort` method that sorts it in place. If you don’t want to mess up your list, you can use the `sorted` function, which returns a new list:

# In[ ]:


x = [4, 1, 2, 3]
y = sorted(x)     # y is [1, 2, 3, 4], x is unchanged

print(f"x = {x}")
print(f"y = {y}")

x.sort()          # now x is [1, 2, 3, 4]

print(f"x = {x}")


# By default, `sort` (and `sorted`) sort a list from smallest to largest based on naively comparing the elements to one another.
# 
# If you want elements sorted from largest to smallest, you can specify a `reverse=True` parameter. And instead of comparing the elements themselves, you can compare the results of a function that you specify with `key`:

# In[ ]:


# sort the list by absolute value from largest to smallest
x = sorted([-4, 1, -2, 3], key=abs, reverse=True)  # is [-4, 3, -2, 1]

print(x)


# ## List Comprehensions

# Frequently, you’ll want to transform a list into another list by choosing only certain elements, by transforming elements, or both. The Pythonic way to do this is with *list comprehensions*:

# In[ ]:


even_numbers = [x for x in range(5) if x % 2 == 0]  # [0, 2, 4]
squares      = [x * x for x in range(5)]            # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]        # [0, 4, 16]

print(f"even_numbers = {even_numbers}")
print(f"squares      = {squares}")
print(f"even_squares = {even_squares}")


# You can similarly turn lists into dictionaries or sets:

# In[ ]:


square_dict = {x: x * x for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
square_set  = {x * x for x in [1, -1]}      # {1}

print(f"square_dict = {square_dict}")
print(f"square_set  = {square_set}")


# If you don’t need the value from the list, it’s common to use an underscore as the variable:

# In[ ]:


zeros = [0 for _ in even_numbers]      # has the same length as even_numbers

print(zeros)


# A list comprehension can include multiple `for`s:

# In[ ]:


pairs = [(x, y)
         for x in range(10)
         for y in range(10)]   # 100 pairs (0,0) (0,1) ... (9,8), (9,9)

print(pairs)


# and later `for`s can use the results of earlier ones:

# In[ ]:


increasing_pairs = [(x, y)                       # only pairs with x < y,
                    for x in range(10)           # range(lo, hi) equals
                    for y in range(x + 1, 10)]   # [lo, lo + 1, ..., hi - 1]

print(increasing_pairs)


# ## Automated Testing and assert

# As data scientists, we’ll be writing a lot of code. How can we be confident our code is correct? One way is to use *automated tests*.
# 
# We will be using `assert` statements, which will cause your code to raise an `AssertionError` if your specified condition is not truthy:

# In[ ]:


# try making the condition False
assert 1 + 1 == 2
assert 1 + 1 == 2, "1 + 1 should equal 2 but didn't"


# As you can see in the second case, you can optionally add a message to be printed if the assertion fails.
# 
# It’s not particularly interesting to assert that 1 + 1 = 2. What’s more interesting is to assert that functions you write are doing what you expect them to:

# In[ ]:


def smallest_item(xs):
    return min(xs)  # what if you change it to max(xs)?

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1


# Another less common use is to assert things about inputs to functions:

# In[ ]:


def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)

smallest_item([10, 20, 5, 40])
# smallest_item([])


# ## Object-Oriented Programming

# Like many languages, Python allows you to define classes that encapsulate data and the functions that operate on them. We’ll use them sometimes to make our code cleaner and simpler.
# 
# Here we’ll construct a class representing a “counting clicker,” the sort that is used at the door to track how many people have shown up for a meeting.
# 
# It maintains a `count`, can be `clicked` to increment the count, allows you to `read_count`, and can be `reset` back to zero. (In real life one of these rolls over from 9999 to 0000, but we won’t bother with that.)
# 
# To define a class, you use the `class` keyword and a PascalCase name:

# A class contains zero or more member functions. By convention, each takes a first parameter, `self`, that refers to the particular class instance.
# 
# Normally, a class has a constructor, named `__init__`. It takes whatever parameters you need to construct an instance of your class and does whatever setup you need:

# Notice that the `__init__` method name starts and ends with double underscores. These “magic” methods are sometimes called “dunder” methods (double-UNDERscore, get it?) and represent “special” behaviors.
# 
# Note: Class methods whose names start with an underscore are—by convention—considered “private,” and users of the class are not supposed to directly call them. However, Python will not stop users from calling them.
# 
# Another such method is `__repr__`, which produces the string representation of a class instance:

# And finally we need to implement the public API of our class:

# In[ ]:


class CountingClicker:
    """A class can/should have a docstring, just like a function"""
    def __init__(self, count = 0):
        self.count = count
        
    def __repr__(self):
        return f"CountingClicker(count={self.count})"
    
    def click(self, num_times = 1):
        """Click the clicker some number of times."""
        self.count += num_times

    def read(self):
        return self.count

    def reset(self):
        self.count = 0


# Having defined it, let’s use `assert` to write some test cases for our clicker:

# In[ ]:


clicker = CountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"
clicker.click()
clicker.click()
assert clicker.read() == 2, "after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, clicker should be back to 0"

print(clicker)


# We’ll also occasionally create subclasses that inherit some of their functionality from a parent class. For example, we could create a non-reset-able clicker by using Counting Clicker as the base class and overriding the reset method to do nothing:

# In[ ]:


# A subclass inherits all the behavior of its parent class.
class NoResetClicker(CountingClicker):
    # This class has all the same methods as CountingClicker

    # Except that it has a reset method that does nothing.
    def reset(self):
        pass


# In[ ]:


clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.reset()
assert clicker2.read() == 1, "reset shouldn't do anything"


# ## Iterables and Generators

# One nice thing about a list is that you can retrieve specific elements by their indices. But you don’t always need this! A list of a billion numbers takes up a lot of memory. If you only want the elements one at a time, there’s no good reason to keep them all around. If you only end up needing the first several elements, generating the entire billion is hugely wasteful.
# 
# Often all we need is to iterate over the collection using `for` and `in`. In this case we can create generators, which can be iterated over just like lists but generate their values lazily on demand.
# 
# One way to create generators is with functions and the `yield` operator:

# In[ ]:


def generate_range(n):
    i = 0
    while i < n:
        yield i   # every call to yield produces a value of the generator
        i += 1


# The following loop will consume the `yield`ed values one at a time:

# In[ ]:


for i in generate_range(1e100000000000000000):
    if (i > 10): break
    print(f"i: {i}")


# (In fact, `range` is itself lazy, so there’s no point in doing this.)
# 
# With a generator, you can even create an infinite sequence:

# In[ ]:


def natural_numbers():
    """returns 1, 2, 3, ..."""
    n = 1
    while True:
        yield n
        n += 1


# although you probably shouldn’t iterate over it without using some kind of `break` logic.
# 
# A second way to create generators is by using `for` comprehensions wrapped in parentheses.
# 
# Note: **you can only iterate through a generator once**. If you need to iterate through something multiple times, you’ll need to either re-create the generator each time or use a list.

# In[ ]:


evens_below_20      = (i for i in range(20) if i % 2 == 0)
evens_below_20_list = [i for i in range(20) if i % 2 == 0]

print(evens_below_20)
print(evens_below_20_list)

for i in evens_below_20:    # try to use a list instead
    print(f"i = {i}")
    
print("try again")
for i in evens_below_20:    # try to use a list instead
    print(f"i = {i}")


# Such a “generator comprehension” doesn’t do any work until you iterate over it (using for or next). We can use this to build up elaborate data-processing pipelines:

# In[ ]:


# None of these computations *does* anything until we iterate
data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_squares = (x ** 2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)
# and so on

assert next(even_squares_ending_in_six) == 16
assert next(even_squares_ending_in_six) == 36
assert next(even_squares_ending_in_six) == 196


# Not infrequently, when we’re iterating over a list or a generator we’ll want not just the values but also their indices. For this common case Python provides an `enumerate` function, which turns values into pairs `(index, value)`:

# In[ ]:


names = ["Alice", "Bob", "Charlie", "Debbie"]

# not Pythonic
for i in range(len(names)):
    print(f"name {i} is {names[i]}")

# also not Pythonic
i = 0
for name in names:
    print(f"name {i} is {names[i]}")
    i += 1

# Pythonic
for i, name in enumerate(names):
    print(f"name {i} is {name}")


# ## Randomness

# As we learn data science, we will frequently need to generate random numbers, which we can do with the `random` module:

# In[ ]:


import random
random.seed(10)  # this ensures we get the same results every time

four_uniform_randoms = [random.random() for _ in range(4)]

# [0.5714025946899135,       # random.random() produces numbers
#  0.4288890546751146,       # uniformly between 0 and 1
#  0.5780913011344704,       # it's the random function we'll use
#  0.20609823213950174]      # most often

print(four_uniform_randoms)


# The random module actually produces pseudorandom (that is, deterministic) numbers based on an internal state that you can set with random.seed if you want to get reproducible results:

# In[ ]:


random.seed(10)         # set the seed to 10
print(random.random())  # 0.57140259469
random.seed(10)         # reset the seed to 10
print(random.random())  # 0.57140259469 again


# We’ll sometimes use `random.randrange`, which takes either one or two arguments and returns an element chosen randomly from the corresponding range:

# In[ ]:


print(random.randrange(10))    # choose randomly from range(10) = [0, 1, ..., 9]
print(random.randrange(3, 6))  # choose randomly from range(3, 6) = [3, 4, 5]


# There are a few more methods that we’ll sometimes find convenient. For example, `random.shuffle` randomly reorders the elements of a list:

# In[ ]:


up_to_ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random.shuffle(up_to_ten)
print(up_to_ten)


# If you need to randomly pick one element from a list, you can use `random.choice`:

# In[ ]:


my_best_friend = random.choice(["Alice", "Bob", "Charlie"])

print(my_best_friend)


# And if you need to randomly choose a sample of elements without replacement (i.e., with no duplicates), you can use  `random.sample`:

# In[ ]:


lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)

print(winning_numbers)


# To choose a sample of elements with replacement (i.e., allowing duplicates), you can just make multiple calls to `random.choice`:

# In[ ]:


four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)


# ## Regular Expressions

# Regular expressions provide a way of searching text. They are incredibly useful, but also fairly complicated—so much so that there are entire books written about them.
# 
# Here are a few examples of how to use them in Python:

# In[ ]:


import re

re_examples = [                        # all of these are true, because
    not re.match("a", "cat"),              #  'cat' doesn't start with 'a'
    re.search("a", "cat"),                 #  'cat' has an 'a' in it
    not re.search("c", "dog"),             #  'dog' doesn't have a 'c' in it
    3 == len(re.split("[ab]", "carbs")),   #  split on a or b to ['c','r','s']
    "R-D-" == re.sub("[0-9]", "-", "R2D2") #  replace digits with dashes
    ]

assert all(re_examples), "all the regex examples should be True"


# One important thing to note is that `re.match` checks whether the beginning of a string matches a regular expression, while `re.search` checks whether any part of a string matches a regular expression.

# ## zip and Argument Unpacking

# Often we will need to `zip` two or more iterables together. The `zip` function transforms multiple iterables into a single iterable of tuples:

# In[ ]:


list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

# zip is lazy, so you have to do something like the following
[pair for pair in zip(list1, list2)]    # is [('a', 1), ('b', 2), ('c', 3)]


# If the lists are different lengths, `zip` stops as soon as the first list ends.
# 
# You can also “unzip” a list using a strange trick:

# In[ ]:


pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)

print(letters)
print(numbers)


# The asterisk (\*) performs *argument unpacking*, which uses the elements of pairs as individual arguments to `zip`. It ends up the same as if you’d called:

# In[ ]:


letters, numbers = zip(('a', 1), ('b', 2), ('c', 3))

print(letters)
print(numbers)


# It may help to mentally picture *argument unpacking* as removing the container the elements are stored in and then immediately pass the individual elements as arguments to another function.
# 
# You can use *argument unpacking* with any function:

# In[ ]:


def add(a, b): return a + b

print(add(1, 2))      # returns 3
try:
    add([1, 2])
except TypeError:
    print("add expects two inputs")
print(add(*[1, 2]))   # returns 3

