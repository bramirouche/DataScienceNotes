#!/usr/bin/env python
# coding: utf-8

# # CSCI 3360 Homework 1

# ## Instructions

# The first homework assignment is designed to get yourself familiar with the Python programming lanaguage. You are to complete this assignment individually, but you may do general searches on the Internet if you are unsure how to complete a certain task using Python (be sure to cite the source of your help). The skeleton code and some sample test cases are provided for you. You are to implement the functions and thoroughly test your code with more test cases. You may assume that the given input parameters are of the correct type, but you may still need to check the validity of the input. You are not allowed to modify the names and parameters of functions. You may choose to write additional helper functions if needed.
# 
# When you are finished, please submit a **Python script file (.py extension) named hw1.py** to eLC before the deadline.
# 
# If you are using Jupyter Notebook, please click **File -> Download as -> Python (.py)** to generate the Python script file. Do **NOT** submit the Jupyter Notebook file (.ipynb extension). Please be sure to excute the Python script file in the command line/console/shell/terminal to make sure your code still runs as expected.

# ## Academic Honesty Statement

# Please first complete the Academic Honesty Statement by filling in your name below. **Any homework submission without the Academic Honesty Statement properly filled in will receive a grade of 0**.

# In[626]:


# first fill in your name
first_name = "Brandon"
last_name  = "Amirouche"

print("****************************************************************")
print("CSCI 3360 Homework 1")
print(f"completed by {first_name} {last_name}")
print(f"""
I, {first_name} {last_name}, certify that the following code
represents my own work. I have neither received nor given 
inappropriate assistance. I have not consulted with another
individual, other than a member of the teaching staff, regarding
the specific problems in this homework. I recognize that any 
unauthorized assistance or plagiarism will be handled in 
accordance with the University of Georgia's Academic Honesty
Policy and the policies of this course.
""")
print("****************************************************************")


# ## Problem 1

# You will be given a string containing only the characters `():`
# 
# Implement the function that returns a number based on the number of sad and smiley faces there are.
# - The happy faces `:)` and `(:` are worth 1.
# - The sad faces `:(` and `):` are worth -1.
# 
# For example, `happy_number(':):(') -> -1`
# - The first 2 characters are `:)`        +1      (Total: 1)
# - The next 2 are `):`                    -1      (Total: 0)
# - The last 2 are `:(`                    -1      (Total: -1)

# In[627]:


# TODO: implement this function
import re
from collections import Counter

def happy_number (s):
    
    q = ":)"
    happy1 = s.count(q)
    w = "(:"
    happy2 = s.count(w)
    countSmiles = happy1 + happy2
    
    e = ":("
    sad1 = s.count(e)
    r = "):"
    sad2 = s.count(r)
    countFrowns = sad1 + sad2
    
    overallFaceCount = countSmiles - countFrowns
   
    return overallFaceCount


# In[628]:


# Sample test cases. You should write more test cases to thoroughly test your code
assert happy_number(':):(') == -1
assert happy_number('(:)') == 2
assert happy_number('::::') == 0
#print(happy_number((':):(')))
#s = ":):(:("
#e = ":("
#s.count(e)
#print(s.count(e))
print(happy_number(':):('))
print(happy_number('(:)'))
print(happy_number('::::'))


# ## Problem 2

# Using list comprehensions, implement the function that returns all even numbers from 0 until the given number (exclusive). Return an empty list if the input parameter <= 0.

# In[629]:


def find_even_nums (n):
    even_in_range = [x for x in range(n) if x % 2 == 0]
    return even_in_range


# In[630]:


# Sample test cases. You should write more test cases to thoroughly test your code
assert find_even_nums(8) == [0, 2, 4, 6]
assert find_even_nums(5) == [0, 2, 4]
assert find_even_nums(2) == [0]
assert find_even_nums(0) == []
print (find_even_nums(8))
print (find_even_nums(5))
print (find_even_nums(2))
print (find_even_nums(0))


# ## Problem 3

# Implement the function that takes a string and returns the number of vowel letters contained within it. Only consider 'a', 'e', 'i', 'o', and 'u' as vowels (not 'y').
# 

# In[631]:


# TODO: implement this function
#3 == len(re.split("[ab]", "carbs")),   #  split on a or b to ['c','r','s']
def count_vowels (s):
    allVowels = re.findall("[aeiou]", s)
    return len(allVowels)


# In[632]:


# Sample test cases. You should write more test cases to thoroughly test your code

assert count_vowels("Celebration") == 5
print(count_vowels("Celebration"))
assert count_vowels("Palm") == 1
print(count_vowels("Palm"))
assert count_vowels("Prediction") == 4
print(count_vowels("Prediction"))


# ## Problem 4

# Implement the function that returns the prime factorization of an integer as a sorted list of tuples. Include the multiplicity of each prime in the tuples:
# - [(prime_0, mult_0), ... , (prime_k, mult_k)]
# - where prime_0 < prime_1 < ... < prime_k
# 
# Please note that 1 is not prime. All inputs will be < 10000.

# In[633]:


# TODO: implement this function
import math
def factorize (n):
    x = []

    for i in range(2, n):
        if(n % i == 0):
            primeBoolean = 1
            
            for y in range(2, (i //2 + 1)):
                if(i % y == 0):
                    primeBoolean = 0
                    break
            
            if (primeBoolean == 1):
                x.append(i)
                print(x)
                


# In[634]:


# Sample test cases. You should write more test cases to thoroughly test your code
#assert factorize(1)  == []
#assert factorize(2)  == [(2, 1)]
#assert factorize(4)  == [(2, 2)]
#assert factorize(10) == [(2, 1), (5, 1)]
#assert factorize(60) == [(2, 2), (3, 1), (5, 1)]
print(factorize(60))


# ## Problem 5

# Implement the function that takes:
# 1. A list of keys.
# 2. A list of values.
# 3. `True`, if key and value should be swapped, else `False`.
# 
# The function returns the constructed dictionary. If input lists are empty or of different sizes, return an empty dictionary. You may assume that keys and values only contain unique values.

# In[635]:


# TODO: implement this function
from collections import defaultdict
def swap_d (keys, values, swap):
    x = {}
    if swap == False:
        zip_ = zip(keys, values)
        x = dict(zip_)
    
    if swap == True:
        zip_ = zip(values, keys)
        x = dict(zip_)
    
    return x
    #keyList, valuesList, _ = zip(*(keys, values))
    #return (print(keyList))


# In[636]:


# Sample test cases. You should write more test cases to thoroughly test your code
assert swap_d ([], [1], False) == {}
assert swap_d ([1, 2, 3], ["one", "two", "three"], False) == { 1: "one", 2: "two", 3: "three" }
assert swap_d ([1, 2, 3], ["one", "two", "three"], True) == { "one": 1, "two": 2, "three": 3 }
assert swap_d (["Paris", 3, 4.5], ["France", "is odd", "is half of 9"], True) == { "France": "Paris", "is odd": 3, "is half of 9": 4.5 }
print (swap_d ([1, 2, 3], ["one", "two", "three"], False))
print (swap_d ([1, 2, 3], ["one", "two", "three"], True))


# ## Problem 6

# Given an integer `limit` being the upper limit of the range of interest, implement the function that returns the last 15 (or however many if there are not enough) palindromes numbers `<= limit` as a list sorted ascendingly. 
# 
# 

# In[637]:


# TODO: implement this function

#this method places the number as a string, flips the string, and returns if it matches with its flipped version (palindrome)
def isItAPalindrome (n):
    n = str(n)
    return n == n[::-1]

def generate_palindromes(limit):
    count = 0
    List = []
    for i in reversed(range(limit+1)):
        if (count < 15):
            if isItAPalindrome(i):
                List.append(i)
                count +=1
        else:
            break
    
    List.sort()

    return List


# In[638]:


# Sample test cases. You should write more test cases to thoroughly test your code
assert generate_palindromes(30) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22]

assert generate_palindromes(151) == [11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111, 121, 131, 141, 151]

assert generate_palindromes(600) == [454, 464, 474, 484, 494, 505, 515, 525, 535, 545, 555, 565, 575, 585, 595]

assert generate_palindromes(999999) == [985589, 986689, 987789, 988889, 989989,
                                        990099, 991199, 992299, 993399, 994499,
                                        995599, 996699, 997799, 998899, 999999]
print (generate_palindromes(30))
print (generate_palindromes(151))
print (generate_palindromes(600))
print (generate_palindromes(999999))
print (isPalindrome(9))


# ## Problem 7

# Implement the function that generates a bar chart of the popularity of programming Languages and saves it as "popularity.png" to the current directory.
# - The bars should be green
# - The title of the bar chart should be "Popularity of Programming Languages"
# - The x-axis label should be "Programming Languages"
# - The y-axis label should be "Popularity"

# In[639]:


import matplotlib.pyplot as plt

# TODO: implement this function
def bar_chart (n, x):
    plt.bar(n, x, color='g',)
    plt.title("Popularity of Programming Languages")
    plt.xlabel("Programming Languages")
    plt.ylabel("Popularity")
    
    
    plt.savefig('popularity.png')
    


# In[640]:


# Sample test cases. You should write more test cases to to generate bar charts with different data
languages = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]
bar_chart (languages, popularity)
# Now check your current directory to see the bar chart


# In[641]:


# be sure clear each plot before generating the next one
plt.gca().clear()


# ## Problem 8

# Implement the function that generates a scatter plot comparing students' final exam scores in Math and Science and saves it as "scores.png" to the current directory.
# - The dots should be blue
# - The title of the scatter plot should be "Math Scores vs. Science Scores"
# - The x-axis label should be "Math Scores"
# - The y-axis label should be "Science Scores"

# In[642]:


# TODO: implement this function
def scatter_plot (n, x):
    # generates and customize your bar chart here
    plt.scatter(n, x, color= 'b')
    plt.title("Math Scores vs. Science Scores")
    plt.axis("equal")
    plt.xlabel("Math Scores")
    plt.ylabel("Science Scores")
    
    plt.savefig('scores.png')
    


# In[643]:


# Sample test cases. You should write more test cases to to generate scatter plots with different data
math    = [88, 92, 80, 89, 100, 80, 60, 100, 80, 34]
science = [35, 79, 79, 48, 100, 88, 32, 45, 20, 30]
scatter_plot (math, science)


# ## Problem 9

# Recall the matrix transpose operation discussed in class. Now implement the function that returns the transpose of a matrix (represented by a 2-dimensional list).

# In[644]:


# Importing typing annotations for matrix
from typing import List

Matrix = List[List[float]]


# In[655]:


# TODO: implement this function
def transpose (A: Matrix) -> Matrix:
    #for n in range(len(A)):
   # iterate through columns
        #for x in range(A[0]):
            #iterating through rows "n" and columns "x" 
            #[n][x] = A[n][x]
        
            #return([]) 
        q = [[A[x][n] for x in range(len(A))]              for n in range(len(A[0]))]

        return q


# In[677]:


# Sample test cases. You should write more test cases to thoroughly test your code
A = [[1, 2, 3],
     [4, 5, 6]]

At = [[1, 4],
      [2, 5],
      [3, 6]]

assert transpose (A) == At
assert transpose (At) == A

I = [[1, 0],
     [0, 1]]

assert transpose (I) == I

print (transpose(A))
print (transpose (At))
print (A)
print (At)


# ## Problem 10

# Recall the matrix multiplication operation discussed in class. Now implement the function that returns the product of two matrices. if the dimensions of the input matrices are incompatible, print an error message and return `None`.

# In[675]:


# TODO: implement this function
def matrix_mult (A: Matrix, B: Matrix) -> Matrix:
    try:
        q = [[sum(a*b for a,b in zip(firstMatrix_row, secondMatrix_col))                    for secondMatrix_col in zip(*B)]                    for firstMatrix_row in A]
    except TypeError:
        print("none")
        
    return q


# In[676]:


# Sample test cases. You should write more test cases to thoroughly test your code
A_At = [[14, 32],
        [32, 77]]

At_A = [[17, 22, 27],
        [22, 29, 36],
        [27, 36, 45]]

testCase1 = [[14, 32, 14, 32],
            [32, 77, 14, 32],
            [14, 32, 14, 32],
            [32, 77, 14, 32],
            [32, 77, 14, 32]]

assert matrix_mult (A, At) == A_At
assert matrix_mult (At, A) == At_A

assert matrix_mult (I, A)  == A
assert matrix_mult (At, I) == At

print (matrix_mult(A, At))
print (matrix_mult (At, A))
print (matrix_mult (I, A))
print (matrix_mult (At, I))
print (matrix_mult (A_At, testCase1))


# In[670]:


print("Cited Sources")
print("https://www.tutorialgateway.org/python-program-to-find-prime-factors-of-a-number/")
print("https://stackoverflow.com/questions/16344284/how-to-generate-a-list-of-palindrome-numbers-within-a-given-range")
print("https://www.geeksforgeeks.org/sort-in-python/")
print("https://www.programiz.com/python-programming/examples/transpose-matrix")
print("https://www.programiz.com/python-programming/examples/multiply-matrix")


# In[ ]:




