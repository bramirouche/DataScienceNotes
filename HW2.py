#!/usr/bin/env python
# coding: utf-8

# # CSCI 3360 Homework 2 - Brandon Amirouche

# ## Instructions

# Consider  the  Iris  dataset  (https://archive.ics.uci.edu/ml/datasets/iris)  and  answer  the  following questions.
# 1.  How many data samples/instances/rows does the Iris dataset contain?
# 
# 2.  How many columns does the dataset contain?  What does each column represent?
# 
# 3.  For each column that contains numeric data, obtain the following: 
# (a)minimum value  (b)maximum value  (c)median  (d)mean  (e)variance  (f)standard deviation
# 
# 4.  For all possible pairs of numeric data columns, obtain the following:
#     (a)covariance matrix  (b)correlation matrix
# 
# 5.  Based on your correlation matrix in 4 (b), which pair of distinct data columns has the strongest positive correlation?  
#     Which pair has the strongest negative correlation?  
#     What do strong positive and negative correlations actually mean?  
#     What do correlations near 0 mean?
# 
# 6.  Now separate the data samples by different iris species.  
#     For each iris species, create a histogram foreach numeric data column of a given iris species.
# 
# 7.  For each iris species, and for each numeric data column of a given iris species, obtain the following:
#     (a)minimum value  (b)maximum value  (c)median  (d)mean  (e)variance  (f)standard deviation
# 
# 8.  For each iris species, and for all possible pairs of numeric data columns of a given iris species, obtainthe following:
#     (a)  covariance matrix(b)  correlation matrix
# 
# 9.  Based on your correlation matrices in 8 (b), describe their similarities and differences. 
#     In particular, highlight any pair of data columns/features that exhibit strong correlations
#     in some iris species but weak  correlations  in  others. Now compare with the correlation matrix  
#     that you obtained in 4 (b).
#     What knowledge/information was hidden by the correlation matrix in 4 (b) when no distinctions ofthe iris species were made?
#     
# 10. Based on the various statistics and plots that you have obtained so far, which data column(s)/feature(s)
#     can be used to distinguish di erent iris species?  Explain how.
# 

# ## Academic Honesty Statement

# In[3]:


# first fill in your name
first_name = "Brandon"
last_name  = "Amirouche"

print("****************************************************************")
print("CSCI 3360 Homework 2")
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

# 1.  How many data samples/instances/rows does the Iris dataset contain?

# The iris dataset contains 150 data samples/instances/rows

# ## Problem 2

# 2.  How many columns does the dataset contain?  What does each column represent?

# There are 5 columns. Each column represents the 5 attributes including:
#     sepal length in cm, 
#     sepal width in cm, 
#     petal length in cm, 
#     petal width in cm, 
#     classification:
#         -- Iris Setosa
#         -- Iris Versicolour
#         -- Iris Virginica

# ## Problem 3

# 3.  For each column that contains numeric data, obtain the following: 
# (a)minimum value  (b)maximum value  (c)median  (d)mean  (e)variance  (f)standard deviation

# In[4]:


import pandas as pd
import numpy as np
header_list = ['sepal length in cm', 'sepal width in cm','petal length in cm', 'petal width in cm', 'classification']
data = pd.read_csv('iris.data', names=header_list)
df = pd.DataFrame(data)
#print(df.columns)

#df.rename(columns = {'Code':'Code-Name',  'Weight':'Weight in kgs'},  inplace = True) 
#df.rename(columns={df()[1]: "attribute", "B": "c"})
#df.columns ['sepal length in cm', 'sepal width in cm','petal length in cm', 'petal width in cm', 'classification']
#clmn = list(df) 
#df = pd.DataFrame(df, columns = ['a', 'b', 'c', 'd', 'e'])
print(f"(a)\n")
print(f"\tThe Minimum value of the Iris dataset by Column:")
print(f"\tFirst column minimum value = {df.min()[0]}")
print(f"\tSecond column minimum value = {df.min()[1]}")
print(f"\tThird column minimum value = {df.min()[2]}")
print(f"\tFourth column minimum value = {df.min()[3]}\n")

print(f"(b)\n")
print(f"\tThe Maximum value of the Iris dataset by Column:")
print(f"\tFirst column maximum value = {df.max()[0]}")
print(f"\tSecond column maximum value = {df.max()[1]}")
print(f"\tThird column maximum value = {df.max()[2]}")
print(f"\tFourth column maximum value = {df.max()[3]}\n")

print(f"(c)\n")
print(f"\tThe Median value of the Iris dataset by Column:")
print(f"\tFirst column median value = {df.median(axis=0)[0]}")
print(f"\tSecond column median value = {df.median(axis=0)[1]}")
print(f"\tThird column median value = {df.median(axis=0)[2]}")
print(f"\tFourth column median value = {df.median(axis=0)[3]}")

print(f"(d)\n")
print(f"\tThe Mean value of the Iris dataset by Column:")
print(f"\tFirst column mean value = {df.mean(axis=0)[0]}")
print(f"\tSecond column mean value = {df.mean(axis=0)[1]}")
print(f"\tThird column mean value = {df.mean(axis=0)[2]}")
print(f"\tFourth column mean value = {df.mean(axis=0)[3]}")

print(f"(e)\n")
print(f"\tThe Variance value of the Iris dataset by Column:")
print(f"\tFirst column variance value = {df.var(axis=0)[0]}")
print(f"\tSecond column variance value = {df.var(axis=0)[1]}")
print(f"\tThird column variance value = {df.var(axis=0)[2]}")
print(f"\tFourth column variance value = {df.var(axis=0)[3]}")

print(f"(f)\n")
print(f"\tThe Standard Deviation value of the Iris dataset by Column:")
print(f"\tFirst column standard deviation = {df.std(axis=0)[0]}")
print(f"\tSecond column standard deviation = {df.std(axis=0)[1]}")
print(f"\tThird column standard deviation = {df.std(axis=0)[2]}")
print(f"\tFourth column standard deviation = {df.std(axis=0)[3]}")


#print(f"First column minimum value = {df.min()[4]}")

#print(df.min()[1]) 

#print(df)


# ## Problem 4

# 4.  For all possible pairs of numeric data columns, obtain the following:
#     (a)covariance matrix  (b)correlation matrix

# In[5]:


print(f"(a)\n")
print(f"The covariance matrix of the Iris Dataframe:")
print(df.cov(min_periods=None))
#def covariance(xs: List[float], ys: List[float]) -> float:
 #   assert len(xs) == len(ys), "xs and ys must have same number of elements"
print(f"(b)\n")
print(f"The correlation matrix of the Iris Dataframe:")
print(df.corr(min_periods=None))


# ## Problem 5

# 5.  Based on your correlation matrix in 4 (b), which pair of distinct data columns has the strongest positive correlation?  
#     Which pair has the strongest negative correlation?  
#     What do strong positive and negative correlations actually mean?  
#     What do correlations near 0 mean?

# (a) The pair of columns regarding petal length and petal width (both in cm) have the strongest positive correlation (negating the perfect 1 that appears when a column is being compared to itself).
# 
# (b) The pair of columns regarding petal length and sepal width (both in cm) have the strongest negative correlation.
# 
# (c) A positive correlation means that the relationship between the two are in unison of appearance. This means that if they have a strong relationship (close to 1) then the more of one columns measured appearance, the more of the other columns measured appearance (i.e. the larger the petal width the larger the petal length). A negative correlation means that the two measured column attributes have an inversed relationship. This means that if they have a strong relationship (close to -1) then the more of one columns measured appearance, the less of the other columns measured appearance (i.e. the larger the sepal width the smaller the petal length).
# 
# (d) A correlation near 0 means that it is a weak relationship. This means that the two columns of measure being compared have nearly no correlation at all.

# ## Problem 6

# 6.  Now separate the data samples by different iris species.  
#     For each iris species, create a histogram for each numeric data column of a given iris species.

# In[6]:


setosa_class = df[df['classification'] == "Iris-setosa"]
versicolor_class = df[df['classification'] == "Iris-versicolor"]
virginica_class = df[df['classification'] == "Iris-virginica"]
setosa_df = pd.DataFrame(setosa_class)
versicolor_df = pd.DataFrame(versicolor_class)
virginica_df = pd.DataFrame(virginica_class)


# In[7]:


print(f"\tSetosa Histogram:\n")
print(setosa_df.hist())


# In[8]:


print(f"\tVersicolor Histogram:\n")
print(versicolor_df.hist())


# In[9]:


print(f"\tVirginica Histogram:\n")
print(virginica_df.hist())


# ## Problem 7

# 7.  For each iris species, and for each numeric data column of a given iris species, obtain the following:
#     (a)minimum value  (b)maximum value  (c)median  (d)mean  (e)variance  (f)standard deviation

# In[10]:


#print(setosa_df)
#min_setosa = setosa_class['sepal length in cm'].min()

print(f"(a)\n")
#Minimum Setosa
print(f"\tThe Minimum value of Setosa by Column:")
print(f"\t\tFirst column minimum value = {setosa_df.min()[0]}")
print(f"\t\tSecond column minimum value = {setosa_df.min()[1]}")
print(f"\t\tThird column minimum value = {setosa_df.min()[2]}")
print(f"\t\tFourth column minimum value = {setosa_df.min()[3]}\n")

#Minimum Versicolor
print(f"\tThe Minimum value of Versicolor by Column:")
print(f"\t\tFirst column minimum value = {versicolor_df.min()[0]}")
print(f"\t\tSecond column minimum value = {versicolor_df.min()[1]}")
print(f"\t\tThird column minimum value = {versicolor_df.min()[2]}")
print(f"\t\tFourth column minimum value = {versicolor_df.min()[3]}\n")

#Minimum Virginica
print(f"\tThe Minimum value of Virginica by Column:")
print(f"\t\tFirst column minimum value = {virginica_df.min()[0]}")
print(f"\t\tSecond column minimum value = {virginica_df.min()[1]}")
print(f"\t\tThird column minimum value = {virginica_df.min()[2]}")
print(f"\t\tFourth column minimum value = {virginica_df.min()[3]}\n")

print(f"(b)\n")
#Maximum Setosa
print(f"\tThe Maximum value of Setosa by Column:")
print(f"\t\tFirst column maximum value = {setosa_df.max()[0]}")
print(f"\t\tSecond column maximum value = {setosa_df.max()[1]}")
print(f"\t\tThird column maximum value = {setosa_df.max()[2]}")
print(f"\t\tFourth column maximum value = {setosa_df.max()[3]}\n")

#Maximum Versicolor
print(f"\tThe Maximum value of Versicolor by Column:")
print(f"\t\tFirst column maximum value = {versicolor_df.max()[0]}")
print(f"\t\tSecond column maximum value = {versicolor_df.max()[1]}")
print(f"\t\tThird column maximum value = {versicolor_df.max()[2]}")
print(f"\t\tFourth column maximum value = {versicolor_df.max()[3]}\n")

#Maximum Virginica
print(f"\tThe Maximum value of Virginica by Column:")
print(f"\t\tFirst column maximum value = {virginica_df.max()[0]}")
print(f"\t\tSecond column maximum value = {virginica_df.max()[1]}")
print(f"\t\tThird column maximum value = {virginica_df.max()[2]}")
print(f"\t\tFourth column maximum value = {virginica_df.max()[3]}\n")

print(f"(c)\n")
#Median Setosa
print(f"\tThe Median value of Setosa by Column:")
print(f"\t\tFirst column median value = {setosa_df.median(axis=0)[0]}")
print(f"\t\tSecond column median value = {setosa_df.median(axis=0)[1]}")
print(f"\t\tThird column median value = {setosa_df.median(axis=0)[2]}")
print(f"\t\tFourth column median value = {setosa_df.median(axis=0)[3]}\n")

#Median Versicolor
print(f"\tThe Median value of Versicolor by Column:")
print(f"\t\tFirst column median value = {versicolor_df.median(axis=0)[0]}")
print(f"\t\tSecond column median value = {versicolor_df.median(axis=0)[1]}")
print(f"\t\tThird column median value = {versicolor_df.median(axis=0)[2]}")
print(f"\t\tFourth column median value = {versicolor_df.median(axis=0)[3]}\n")

#Median Virginica
print(f"\tThe Median value of Virginica by Column:")
print(f"\t\tFirst column median value = {virginica_df.median(axis=0)[0]}")
print(f"\t\tSecond column median value = {virginica_df.median(axis=0)[1]}")
print(f"\t\tThird column median value = {virginica_df.median(axis=0)[2]}")
print(f"\t\tFourth column median value = {virginica_df.median(axis=0)[3]}\n")

print(f"(d)\n")
#Mean Setosa
print(f"\tThe Mean value of Setosa by Column:")
print(f"\t\tFirst column mean value = {setosa_df.mean(axis=0)[0]}")
print(f"\t\tSecond column mean value = {setosa_df.mean(axis=0)[1]}")
print(f"\t\tThird column mean value = {setosa_df.mean(axis=0)[2]}")
print(f"\t\tFourth column mean value = {setosa_df.mean(axis=0)[3]}\n")

#Mean Versicolor
print(f"\tThe Mean value of Versicolor by Column:")
print(f"\t\tFirst column mean value = {versicolor_df.mean(axis=0)[0]}")
print(f"\t\tSecond column mean value = {versicolor_df.mean(axis=0)[1]}")
print(f"\t\tThird column mean value = {versicolor_df.mean(axis=0)[2]}")
print(f"\t\tFourth column mean value = {versicolor_df.mean(axis=0)[3]}\n")

#Mean Virginica
print(f"\tThe Mean value of Virginica by Column:")
print(f"\t\tFirst column mean value = {virginica_df.mean(axis=0)[0]}")
print(f"\t\tSecond column mean value = {virginica_df.mean(axis=0)[1]}")
print(f"\t\tThird column mean value = {virginica_df.mean(axis=0)[2]}")
print(f"\t\tFourth column mean value = {virginica_df.mean(axis=0)[3]}\n")

print(f"(e)\n")
#Variance Setosa
print(f"\tThe Variance value of Setosa by Column:")
print(f"\t\tFirst column variance value = {setosa_df.var(axis=0)[0]}")
print(f"\t\tSecond column variance value = {setosa_df.var(axis=0)[1]}")
print(f"\t\tThird column variance value = {setosa_df.var(axis=0)[2]}")
print(f"\t\tFourth column variance value = {setosa_df.var(axis=0)[3]}\n")

#Variance Versicolor
print(f"\tThe Variance value of Versicolor by Column:")
print(f"\t\tFirst column variance value = {versicolor_df.var(axis=0)[0]}")
print(f"\t\tSecond column variance value = {versicolor_df.var(axis=0)[1]}")
print(f"\t\tThird column variance value = {versicolor_df.var(axis=0)[2]}")
print(f"\t\tFourth column variance value = {versicolor_df.var(axis=0)[3]}\n")

#Variance Virginica
print(f"\tThe Variance value of Virginica by Column:")
print(f"\t\tFirst column variance value = {virginica_df.var(axis=0)[0]}")
print(f"\t\tSecond column variance value = {virginica_df.var(axis=0)[1]}")
print(f"\t\tThird column variance value = {virginica_df.var(axis=0)[2]}")
print(f"\t\tFourth column variance value = {virginica_df.var(axis=0)[3]}\n")

print(f"(f)\n")
#Standard Deviation Setosa
print(f"\tThe Standard Deviation value of Setosa by Column:")
print(f"\t\tFirst column standard deviation value = {setosa_df.std(axis=0)[0]}")
print(f"\t\tSecond column standard deviation value = {setosa_df.std(axis=0)[1]}")
print(f"\t\tThird column standard deviation value = {setosa_df.std(axis=0)[2]}")
print(f"\t\tFourth column standard deviation value = {setosa_df.std(axis=0)[3]}\n")

#Variance Versicolor
print(f"\tThe Standard Deviation value of Versicolor by Column:")
print(f"\t\tFirst column standard deviation value = {versicolor_df.std(axis=0)[0]}")
print(f"\t\tSecond column standard deviation value = {versicolor_df.std(axis=0)[1]}")
print(f"\t\tThird column standard deviation value = {versicolor_df.std(axis=0)[2]}")
print(f"\t\tFourth column standard deviation value = {versicolor_df.std(axis=0)[3]}\n")

#Variance Virginica
print(f"\tThe Standard Deviation value of Virginica by Column:")
print(f"\t\tFirst column standard deviation value = {virginica_df.std(axis=0)[0]}")
print(f"\t\tSecond column standard deviation value = {virginica_df.std(axis=0)[1]}")
print(f"\t\tThird column standard deviation value = {virginica_df.std(axis=0)[2]}")
print(f"\t\tFourth column standard deviation value = {virginica_df.std(axis=0)[3]}\n")

#min_versicolor = versicolor_class['sepal length in cm'].min()
#print(f"\tMinimum Value of versicolor = {min_versicolor}")

#min_virginica = virginica_class['sepal length in cm'].min()
#print(f"\tMinimum Value of virginica = {min_virginica}")


# ## Problem 8

# 8.  For each iris species, and for all possible pairs of numeric data columns of a given iris species, obtain the following:
#     (a)  covariance matrix(b)  correlation matrix

# In[11]:


print(f"(a)\n")
#setosa_df.cov(min_periods=None)
#versicolor_df.cov(min_periods=None)
#virginica_df.cov(min_periods=None)
print(f"The covariance matrix of the Setosa Dataframe:")
print(setosa_df.cov(min_periods=None))

print(f"\n")
print(f"The covariance matrix of the Versicolor Dataframe:")
print(versicolor_df.cov(min_periods=None))

print(f"\n")
print(f"The covariance matrix of the Virginica Dataframe:")
print(virginica_df.cov(min_periods=None))

print(f"\n")
print(f"\n")



print(f"(b)\n")
print(f"The correlation matrix of the Setosa Dataframe:")
print(setosa_df.corr(min_periods=None))

print(f"\n")
print(f"The correlation matrix of the Versicolor Dataframe:")
print(versicolor_df.corr(min_periods=None))

print(f"\n")
print(f"The correlation matrix of the Virginica Dataframe:")
print(virginica_df.corr(min_periods=None))


# ## Problem 9

# 9.  Based on your correlation matrices in 8 (b), describe their similarities and differences. 
#     In particular, highlight any pair of data columns/features that exhibit strong correlations
#     in some iris species but weak  correlations  in  others. Now compare with the correlation matrix  
#     that you obtained in 4 (b).
#     What knowledge/information was hidden by the correlation matrix in 4 (b) when no distinctions of the iris species were         made?

# The versicolor and Viginica correlation tables both have a relatively high correlation for sepal length and petal length, however the setosa species seems to have a low correlation measured for this comparison. Petal length and petal width have a high correlation within only the versicolor species. 
# 
# 
# Interestingly, petal length and petal width had the highest positive overall correlation in the overall iris dataframe, however, only the versicolor dataframe shows any real possible correlation here. Even within the versicolor dataframe, the correlation is still not as high as in the original dataframe. Also, there seems to be no negative correlations in each of the species dataframes while the overall iris dataframe showed a couple of negative correlations.

# ## Problem 10

# 10. Based on the various statistics and plots that you have obtained so far, which data column(s)/feature(s)
#     can be used to distinguish different iris species?  Explain how.

# Columns and Features that can be used to distinguish different iris species include large petal length and petal width for versicolor species. A low correlation between petal length and sepal length seems to be a sign for Setosa. A large correlation between sepal length and sepal width seems to be a sign that a large measure for both can also be a sign for Setosa. 
