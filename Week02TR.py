#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra

# Linear algebra is the branch of mathematics that deals with *vector spaces*. There are entire courses and textbooks on this topic. A large number of data science concepts and techniques rely on linear algebra. We will be doing a brief introduction on the basic elements of linear algebra here.

# ## Vectors

# Abstractly, *vectors* are objects that can be added together to form new vectors and that can be multiplied by *scalars* (i.e., numbers), also to form new vectors.
# 
# Concretely (for us), vectors are points in some finite-dimensional space (e.g., (x, y) coordinates in 2D space, (x, y, z) coordinates in 3D space). Although you might not think of your data as vectors, they are often a useful way to represent numeric data.
# 
# Here are some more examples:
# - if you have the heights, weights, and ages of a large number of people, you can treat your data as 3D vectors `[height, weight, age]`.
# - If you’re teaching a class with four exams, you can treat student grades as 4D vectors `[exam1, exam2, exam3, exam4]`.
# 
# The simplest from-scratch approach is to represent vectors as lists of numbers. A list of three numbers corresponds to a vector in three-dimensional space, and vice versa.
# 
# We’ll accomplish this with a type alias that says a `Vector` is just a list of `float`s.
# 
# Note: for illustrative and educational purposes, we will implement a couple of `Vector` operations here. There are plenty of well written linear algebra packages that we will actually be using later (e.g., [NumPy](https://numpy.org/)).

# In[ ]:


from typing import List

Vector = List[float]

height_weight_age = [70,  # inches,
                     170, # pounds,
                     40 ] # years

grades = [95,   # exam1
          80,   # exam2
          75,   # exam3
          62 ]  # exam4


# We’ll also want to perform *arithmetic* on vectors. Because Python lists aren’t vectors (and hence provide no facilities for vector arithmetic), we’ll need to build these arithmetic tools ourselves. So let’s start with that.
# 
# To begin with, we’ll frequently need to add two vectors. Vectors add *componentwise*. This means that if two vectors `v` and `w` are the same length, their sum is just the vector whose first element is `v[0] + w[0]`, whose second element is `v[1] + w[1]`, and so on. (If they’re not the same length, then we’re not allowed to add them.)
# 
# For example, adding the vectors `[1, 2]` and `[2, 1]` results in `[1 + 2, 2 + 1]` or `[3, 3]`.
# 
# We can easily implement this by `zip`-ing the vectors together and using a list comprehension to add the corresponding elements:

# In[ ]:


# The : Vector and -> Vector portions of the code are referred to as "Type Annotations". They are mostly for
# documentation/readability purposes. Some IDEs and type checkers can also use these to check your code to
# ensure that all typing requirements are satisfied.
def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

print(add([1, 2, 3], [4, 5, 6]))
# print([1, 2, 3] + [4, 5, 6]) not vector addition


# Similarly, to subtract two vectors we just subtract the corresponding elements:

# In[ ]:


def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]

print(subtract([5, 7, 9], [4, 5, 6]))


# We’ll also sometimes want to componentwise sum a list of vectors—that is, create a new vector whose first element is the sum of all the first elements, whose second element is the sum of all the second elements, and so on:

# In[ ]:


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    # recall that all() returns True if all the items in the iterable are Truthy
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    
    # the i-th element of the result is the sum of every vector[i]
    # the i-loop is the outer loop, the inner loop uses a fixed i and generate the sum of all vector elements at index i
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

print(vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]))


# We’ll also need to be able to multiply a vector by a scalar, which we do simply by multiplying each element of the vector by that number:

# In[ ]:


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

print(scalar_multiply(2, [1, 2, 3]))


# This allows us to compute the componentwise means of a list of (same-sized) vectors:

# In[ ]:


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

print(vector_mean([[1, 2],
                   [3, 4],
                   [5, 6]]))


# A less obvious tool is the *dot product*. The dot product of two vectors is the sum of their componentwise products:

# In[ ]:


def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

print(dot([1, 2, 3], [4, 5, 6])) # 32 = 1 * 4 + 2 * 5 + 3 * 6


# Using this, it’s easy to compute a vector’s *sum of squares*:

# In[ ]:


def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

print(sum_of_squares([1, 2, 3])) # 14 = 1 * 1 + 2 * 2 + 3 * 3


# which we can use to compute its *magnitude* (or length):

# In[ ]:


import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function

print(magnitude([3, 4])) # 5 = sqrt(3^2 + 4^2)


# We now have all the pieces we need to compute the distance between two vectors, defined as:
# 
# $$ \sqrt{(v_1 - w_1)^2 + \dots + (v_n - w_n)^2}$$
# 
# Note that we are computing distance between two 2D vectors, then it's just analogous to computing the distance between two points in the Euclidean space, also known as the Manhattan distance. The vector distance formula above may be considered as higher-dimensional generalizations of Manhattan distance.
# 
# In code:

# In[ ]:


def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))

print(distance([1, 1], [4, 5]))


# This is possibly clearer if we write it as (the equivalent):

# In[ ]:


def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))

print(distance([1, 1], [4, 5]))


# ## Matrices

# A *matrix* is a two-dimensional collection of numbers. We will represent matrices as lists of lists, with each inner list having the same size and representing a row of the matrix.
# 
# If `A` is a matrix, then `A[i][j]` is the element in the *i*th row and the *j*th column.
# 
# Per mathematical convention, we will frequently use capital letters to represent matrices. For example:

# In[ ]:


# Another type alias
Matrix = List[List[float]]

A = [[1, 2, 3],  # A has 2 rows and 3 columns
     [4, 5, 6]]

B = [[1, 2],     # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]

print(A)
print(B)


# Note: In mathematics, you would usually name the first row of the matrix “row 1” and the first column “column 1.” Because we’re representing matrices with Python lists, which are zero-indexed, we’ll call the first row of a matrix “row 0” and the first column “column 0.”
# 
# Given this list-of-lists representation, the matrix `A` has `len(A)` rows and `len(A[0])` columns, which we consider its `shape`:

# In[ ]:


from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # number of elements in first row
    return num_rows, num_cols

print(shape([[1, 2, 3],
             [4, 5, 6]]))   # 2 rows, 3 columns


# If a matrix has *n* rows and *k* columns, we will refer to it as an *n × k matrix*. We can (and sometimes will) think of each row of an *n × k* matrix as a vector of length *k*, and each column as a vector of length *n*:

# In[ ]:


def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]             # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j]          # jth element of row A_i
            for A_i in A]   # for each row A_i

print(get_row(A, 1))
print(get_column(A, 1))


# We’ll also want to be able to create a matrix given its shape and a function for generating its elements. We can do this using a nested list comprehension:

# In[ ]:


# The 'Callable' type represents a function
from typing import Callable

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)             # given i, create a list
             for j in range(num_cols)]  #   [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # create one list for each i


# Given this function, you could make a 5 × 5 *identity matrix* (with 1s on the diagonal and 0s elsewhere):

# In[ ]:


# equivalent to the annoynomous function (defined by the keyword 'lambda') below
def fun (i: int, j: int):
    return 1 if i == j else 0

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

id5 = identity_matrix(5)
print(id5)
assert id5 == [[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]


# Matrices will be important to us for several reasons.
# 
# First, we can use a matrix to represent a dataset consisting of multiple vectors, simply by considering each vector as a row of the matrix. For example, if you had the heights, weights, and ages of 1,000 people, you could put them in a 1,000 × 3 matrix:

# In[ ]:


data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19],
        # ....
       ]


# Second, as we’ll see later, we can use an *n × k* matrix to represent a linear function that maps k-dimensional vectors to n-dimensional vectors (through matrix-vector multiplication). Several of our techniques and concepts will involve such functions.
# 
# Third, matrices can be used to represent binary relationships. Recall that last week we represented the edges of a social network as a collection of `pairs (i, j)`:

# In[ ]:


friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]


# An alternative representation would be to create a matrix `A` such that `A[i][j]` is 1 if nodes i and j are connected and 0 otherwise.

# In[ ]:


#            user 0  1  2  3  4  5  6  7  8  9
#
friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9


# If there are very few connections, this is a much more inefficient representation, since you end up having to store a lot of zeros.
# 
# However, with the matrix representation it is much quicker to check whether two nodes are connected—you just have to do a matrix lookup instead of (potentially) inspecting every edge:

# In[ ]:


assert friend_matrix[0][2] == 1, "0 and 2 are friends"
assert friend_matrix[0][8] == 0, "0 and 8 are not friends"


# Similarly, to find a node’s connections, you only need to inspect the column (or the row) corresponding to that node:

# In[ ]:


# only need to look at one row
friends_of_five = [i
                   for i, is_friend in enumerate(friend_matrix[5])
                   if is_friend]

print(friends_of_five) # user 4, 6, and 7 are friends with user 5

