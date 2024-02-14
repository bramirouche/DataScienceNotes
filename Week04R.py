#!/usr/bin/env python
# coding: utf-8

# # Introduction to pandas

# Adapted from [Julia Evans](https://jvns.ca/)'s [pandas cookbook](https://github.com/jvns/pandas-cookbook).
# 
# This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/)

# "[Pandas](https://pandas.pydata.org/) *is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.*"
# 
# If you installed Anaconda, pandas should have already been installed. Otherwise, you may need to install it yourself.

# In[ ]:


get_ipython().system('pip install pandas')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Make the graphs a bit prettier, and bigger
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 5) # resize plots to be 15 in by 5 in


# ## Reading data from a csv file

# You can read data from a CSV file using the `read_csv` function. By default, it assumes that the fields are comma-separated.
# 
# We're going to be looking some cyclist data from Montréal. Here's the [original page](http://donnees.ville.montreal.qc.ca/dataset/velos-comptage) (in French). I have already downloaded and place the csv file in the current directory. We will be using the data from 2012.
# 
# This dataset is a list of how many people were on 7 different bike paths in Montreal, each day.

# In[ ]:


broken_df = pd.read_csv('bikes.csv', encoding = "ISO-8859-1")


# In[ ]:


# examine the dataframe
broken_df


# You'll notice that this is totally broken! `read_csv` has a bunch of options that will let us fix that, though. Here we'll
# 
# * change the column separator to a `;`
# * Set the encoding to `'latin1'` (the default is `'utf8'`)
# * Parse the dates in the 'Date' column
# * Tell it that our dates have the day first instead of the month first
# * Set the index to be the 'Date' column

# In[ ]:


fixed_df = pd.read_csv('bikes.csv', sep=';', encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')
fixed_df


# In[ ]:


# Examine data from a specific date range.
# This can be done because we've indicated that the "Date" column should be treated as an index column
fixed_df['2012-01-03':'2012-01-06']


# ## Selecting a column

# When you read a CSV, you get a kind of object called a `DataFrame`, which is made up of rows and columns. You get columns out of a DataFrame the same way you get elements out of a dictionary.
# 
# Here's an example:

# In[ ]:


print(type(fixed_df))
fixed_df['Berri 1']


# ## Plotting a column

# Just add `.plot()` to the end! How could it be easier? =)
# 
# We can see that, unsurprisingly, not many people are biking in January, February, and March, 

# In[ ]:


fixed_df['Berri 1'].plot()


# We can also plot all the columns just as easily. We'll make it a little bigger, too.
# You can see that it's more squished together, but all the bike paths behave basically the same -- if it's a bad day for cyclists, it's a bad day everywhere.

# In[ ]:


fixed_df.plot(figsize=(15, 10))


# # Dealing with Large Datasets

# We're going to use a new dataset here, to demonstrate how to deal with larger datasets. This is a subset of the of 311 service requests from [NYC Open Data](https://nycopendata.socrata.com/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9). 

# In[ ]:


# because of mixed types we specify dtype to prevent any errors
complaints = pd.read_csv('311-service-requests.csv', dtype='object')


# Without specifying the `dtype` (data type for columns) parameter, you might see a warning like "DtypeWarning: Columns (8) have mixed types". This means that it's encountered a problem reading in our data. In this case it almost certainly means that it has columns where some of the entries are strings and some are integers.
# 
# For now we're going to ignore it and hope we don't run into a problem, but in the long run we'd need to investigate this warning.

# ## What's even in it?

# When you print a large dataframe, it will only show you the first few rows.

# In[ ]:


complaints


# ## Selecting columns and rows

# To select a column, we index with the name of the column, like this:

# In[ ]:


complaints['Complaint Type']


# To get the first 5 rows of a dataframe, we can use a slice: `df[:5]`.
# 
# This is a great way to get a sense for what kind of information is in the dataframe -- take a minute to look at the contents and get a feel for this dataset.

# In[ ]:


complaints[:5]


# We can combine these to get the first 5 rows of a column:

# In[ ]:


complaints['Complaint Type'][:5]


# and it doesn't matter which direction we do it in:

# In[ ]:


complaints[:5]['Complaint Type']


# ## Selecting multiple columns

# What if we just want to know the complaint type and the borough, but not the rest of the information? Pandas makes it really easy to select a subset of the columns: just index with list of columns you want.

# In[ ]:


complaints[['Complaint Type', 'Borough']]


# That showed us a summary, and then we can look at the first 10 rows:

# In[ ]:


complaints[['Complaint Type', 'Borough']][:10]


# ## What's the most common complaint type?

# This is a really easy question to answer! There's a `.value_counts()` method that we can use:

# In[ ]:


complaints['Complaint Type'].value_counts()


# If we just wanted the top 10 most common complaints, we can do this:

# In[ ]:


complaint_counts = complaints['Complaint Type'].value_counts()
complaint_counts[:10]


# But it gets better! We can plot them!

# In[ ]:


complaint_counts[:10].plot(kind='bar')


# ## Selecting only noise complaints

# I'd like to know which borough has the most noise complaints. First, let's remind ourselves what the data look like.

# In[ ]:


complaints[:5]


# To get the noise complaints, we need to find the rows where the "Complaint Type" column is "Noise - Street/Sidewalk". I'll show you how to do that, and then explain what's going on.

# In[ ]:


noise_complaints = complaints[complaints['Complaint Type'] == "Noise - Street/Sidewalk"]
noise_complaints[:3]


# If you look at `noise_complaints`, you'll see that this worked, and it only contains complaints with the right complaint type. But how does this work? Let's deconstruct it into two pieces

# In[ ]:


complaints['Complaint Type'] == "Noise - Street/Sidewalk"


# This is a big array of `True`s and `False`s, one for each row in our dataframe. When we index our dataframe with this array, we get just the rows where our boolean array evaluated to `True`.  It's important to note that for row filtering by a boolean array the length of our dataframe's index must be the same length as the boolean array used for filtering.
# 
# You can also combine more than one condition with the `&` operator like this:

# In[ ]:


is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
in_brooklyn = complaints['Borough'] == "BROOKLYN"
complaints[is_noise & in_brooklyn][:5]


# Or if we just wanted a few columns:

# In[ ]:


complaints[is_noise & in_brooklyn][['Complaint Type', 'Borough', 'Created Date', 'Descriptor']][:10]


# ## So, which borough has the most noise complaints?

# In[ ]:


is_noise = complaints['Complaint Type'] == "Noise - Street/Sidewalk"
noise_complaints = complaints[is_noise]
noise_complaints['Borough'].value_counts()


# It's Manhattan! But what if we wanted to divide by the total number of complaints, to make it make a bit more sense? That would be easy too:

# In[ ]:


noise_complaint_counts = noise_complaints['Borough'].value_counts()
complaint_counts = complaints['Borough'].value_counts()


# In[ ]:


noise_complaint_counts / complaint_counts


# In[ ]:


(noise_complaint_counts / complaint_counts).plot(kind='bar')


# So Manhattan really does complain more about noise than the other boroughs!

# ## Adding a column to our dataframe

# Okay! We're going back to our bike path dataset here. I was curious about whether Montreal is more of a commuter city or a biking-for-fun city -- do people bike more on weekends, or on weekdays?
# 
# First, we need to load up the data again.

# In[ ]:


bikes = pd.read_csv('bikes.csv', sep=';', encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')
bikes['Berri 1'].plot()


# Next up, we're just going to look at the Berri bike path. So we're going to create a dataframe with just the Berri bikepath in it.

# In[ ]:


berri_bikes = bikes[['Berri 1']].copy()


# In[ ]:


berri_bikes[:5]


# Next, we need to add a 'weekday' column. Firstly, we can get the weekday from the index.

# In[ ]:


berri_bikes.index


# You can see that actually some of the days are missing -- only 310 days of the year are actually there. Who knows why.
# 
# Pandas has a bunch of really great time series functionality, so if we wanted to get the day of the month for each row, we could do it like this:

# In[ ]:


berri_bikes.index.day


# We actually want the weekday, though:

# In[ ]:


berri_bikes.index.weekday


# These are the days of the week, where 0 is Monday. I found out that 0 was Monday by checking on a calendar.
# 
# Now that we know how to *get* the weekday, we can add it as a column in our dataframe like this:

# In[ ]:


berri_bikes.loc[:,'weekday'] = berri_bikes.index.weekday
berri_bikes[:5]


# ## Adding up the cyclists by weekday

# This turns out to be really easy!
# 
# Dataframes have a `.groupby()` method that is similar to SQL groupby, if you're familiar with that. We are not going to dig into this right now -- if you want to to know more, [the documentation](http://pandas.pydata.org/pandas-docs/stable/groupby.html) is really good.
# 
# In this case, `berri_bikes.groupby('weekday').aggregate(sum)` means "Group the rows by weekday and then add up all the values with the same weekday".

# In[ ]:


weekday_counts = berri_bikes.groupby('weekday').aggregate(sum)
weekday_counts


# It's hard to remember what 0, 1, 2, 3, 4, 5, 6 mean, so we can fix it up and graph it:

# In[ ]:


weekday_counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts


# In[ ]:


weekday_counts.plot(kind='bar')


# So it looks like Montrealers are commuter cyclists -- they bike much more during the week.
