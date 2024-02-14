#!/usr/bin/env python
# coding: utf-8

# # A Crash Course in Python

# ## Whitespace Formatting

# Many languages use curly braces to delimit blocks of code. Python uses **indentation**
# 
# Python considers tabs and spaces **different indentation** and will not be able to run your code if you mix the two.
# 
# When writing Python you should **always use spaces**, never tabs (Jupyter Notebook, by default, converts tabs to spaces. Other editors should be configured to do the same).

# In[ ]:


# The pound sign marks the start of a comment. Python itself
# ignores the comments, but they're helpful for anyone reading the code.
for i in [1, 2, 3, 4, 5]:
    print("i = ", i)                    # first line in "for i" block
    for j in [1, 2, 3, 4, 5]:
        print("j = ", j)                # first line in "for j" block
        print("i + j = ", i + j)        # last line in "for j" block
    print("i = ", i)                    # last line in "for i" block
print("done looping")


# Whitespace is ignored inside parentheses and brackets, which can be helpful for longwinded computations:

# In[ ]:


long_winded_computation = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 +
                           13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

easier_to_read_list_of_lists = [[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]

print("same 2D list:")
print(list_of_lists)
print(easier_to_read_list_of_lists)


# You can also use a backslash to indicate that a statement continues onto the next line, although we’ll rarely do this:

# In[ ]:


two_plus_three = 2 +                  3

print(two_plus_three)


# ## Modules

# Modules not loaded by default can be imported.

# In[ ]:


import platform

x = platform.system()
print(x)


# Alias may also be used, especially for long package names

# In[ ]:


import platform as pl

x = pl.system()
print(x)


# If you need a few specific values from a module, you can import them explicitly and use them without qualification:

# In[ ]:


from platform import system
x = system()
print(x)


# ## Functions

# A function is a rule for taking zero or more inputs and returning a corresponding output.
# 
# In Python, we typically define functions using `def`:

# In[ ]:


def double(x):
    """
    This is where you put an optional docstring that explains what the
    function does. For example, this function multiplies its input by 2.
    """
    return x * 2


# Python functions are *first-class*, which means that we can assign them to variables and pass them into functions just like any other arguments:

# In[ ]:


def apply_to_one(f):
    """Calls the function f with 1 as its argument"""
    return f(1)


# In[ ]:


my_double = double          # refers to the previously defined function
x = apply_to_one(my_double) # equals 2
print(x)


# Function parameters can also be given default arguments, which only need to be specified when you want a value other than the default:

# In[ ]:


def my_print(message = "my default message"):
    print(message)

my_print("hello")   # prints 'hello'
my_print()          # prints 'my default message'


# It is sometimes useful to specify arguments by name:

# In[ ]:


def full_name(first = "What's-his-name", last = "Something"):
    return first + " " + last

print(full_name("Joel", "Grus"))     # "Joel Grus"
print(full_name("Joel"))             # "Joel Something"
print(full_name(last="Grus"))        # "What's-his-name Grus"


# ## Strings

# Strings can be delimited by single or double quotation marks (but the quotes have to match):

# In[ ]:


single_quoted_string = 'data science'
double_quoted_string = "data science"
#invalid_string = 'data science"

print(single_quoted_string)
print(double_quoted_string)


# Python uses backslashes to encode special characters. For example:

# In[ ]:


tab_string = "\t"       # represents the tab character
print("before tab", tab_string, "after tab")
print(len(tab_string))  # is 1


# If you want backslashes as backslashes (which you might in Windows directory names or in regular expressions), you can create raw strings using `r""`:

# In[ ]:


not_tab_string = r"\t"         # represents the characters '\' and 't'
print(not_tab_string)
print(len(not_tab_string))     # is 2


# You can create multiline strings using three double quotes:

# In[ ]:


multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""

print(multi_line_string)


# A new feature since Python 3.6 is the *f-string*, which provides a simple way to substitute values into strings. For example, if we had the first name and last name given separately:

# In[ ]:


first_name = "Joel"
last_name  = "Grus"


# we might want to combine them into a full name. There are multiple ways to construct such a full_name string:

# In[ ]:


full_name1 = first_name + " " + last_name             # string addition
full_name2 = "{0} {1}".format(first_name, last_name)  # string.format
full_name3 = f"{first_name} {last_name}"              # f-string, preferred

print(full_name1)
print(full_name2)
print(full_name3)


# ## Exceptions

# When something goes wrong, Python raises an exception. Unhandled, exceptions will cause your program to crash. You can handle them using try and except:

# In[ ]:


try:
    print(0 / 0)
except ZeroDivisionError:
    print("cannot divide by zero")

