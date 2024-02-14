#!/usr/bin/env python
# coding: utf-8

# # Getting Data

# In order to be a data scientist you need data. In fact, as a data scientist you will spend an embarrassingly large fraction of your time acquiring, cleaning, and transforming data.
# 
# We’ll look at different ways of getting data into Python and into the right formats.

# ## Reading Files

# You can explicitly read from and write to files directly in your code. Python makes working with files pretty simple.
# 
# The first step to working with a text file is to obtain a *file object* using `open`:

# In[ ]:


# 'r' means read-only, it's assumed if you leave it out
file_for_reading = open('reading_file.txt', 'r')
file_for_reading2 = open('reading_file.txt')

# 'w' is write -- will destroy the file if it already exists!
# if the file does not exists, then it will be created.
file_for_writing = open('writing_file.txt', 'w')

# 'a' is append -- for adding to the end of the file
# Just like the 'w' mode above, the file will be created
# if it doesn't already exists.
file_for_appending = open('appending_file.txt', 'a')

# don't forget to close your files when you're done
file_for_reading.close()
file_for_reading2.close()
file_for_writing.close()
file_for_appending.close()


# Because it is easy to forget to close your files, you should always use them in a `with` block, at the end of which they will be closed automatically:

# In[ ]:


import re

starts_with_hash = 0

# the with block ensures automatic and graceful acquisition and release of resources
with open('input.txt') as f:
    for line in f:                  # look at each line in the file
        if re.match("^#",line):     # use a regex to see if it starts with '#'
            starts_with_hash += 1   # if it does, add 1 to the count

print(f"number of lines that starts with # is {starts_with_hash}")


# Every line you get this way ends in a newline character, so you’ll often want to strip it before doing anything with it.
# 
# For example, imagine you have a file full of email addresses, one per line, and you need to generate a histogram of the domains. We will assume the domain is the part of the email addresses that comes after the @.
# 
# Note that in Python 3, Python will, by default, handle different line separator characters from different OS and convert them all into '\n', and vice-versa.

# In[ ]:


# Here we are creating our example file containing some email addresses, one per line
with open('email_addresses.txt', 'w') as f:
    f.write("joelgrus@gmail.com\n")
    f.write("joel@m.datasciencester.com\n")
    f.write("joelgrus@m.datasciencester.com\n")


# In[ ]:


def get_domain(email_address: str) -> str:
    """Split on '@' and return the last piece"""
    return email_address.lower().split("@")[-1]

# a couple of tests
assert get_domain('joelgrus@gmail.com') == 'gmail.com'
assert get_domain('joel@m.datasciencester.com') == 'm.datasciencester.com'

from collections import Counter

with open('email_addresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
                            for line in f
                            if "@" in line)

print(domain_counts)


# ## Delimited Files

# The hypothetical email addresses file we just processed had one address per line. More frequently you’ll work with files with lots of data on each line.
# 
# These files are very often either *comma-separated* or *tab-separated*: each line has several fields, with a comma or a tab indicating where one field ends and the next field starts.
# 
# This starts to get complicated when you have fields with commas and tabs and newlines in them (which you inevitably will). For this reason, you should never try to parse them yourself. Instead, you should use Python’s `csv` module (or the `pandas` library, or some other library that’s designed to read comma-separated or tab delimited files).
# 
# If your file has no headers (which means you probably want each row as a list, and which places the burden on you to know what’s in each column), you can use `csv.reader` to iterate over the rows, each of which will be an appropriately split list.
# 
# For example, if we had a tab-delimited file of stock prices:

# In[ ]:


with open('tab_delimited_stock_prices.txt', 'w') as f:
    f.write("""6/20/2020\tAAPL\t90.91
6/20/2020\tMSFT\t41.68
6/20/2020\tFB\t64.5
6/19/2020\tAAPL\t91.86
6/19/2020\tMSFT\t41.51
6/19/2020\tFB\t64.34
""")


# we could process them with:

# In[ ]:


def process(date: str, symbol: str, closing_price: float) -> None:
    # this function could be redesigned to do other things.
    print(f"The closing price of {symbol} on {date} is ${closing_price}.")

import csv

with open('tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)


# If your file has headers, you can either skip the header row with an initial call to `reader.next`, or get each row as a dict (with the headers as keys) by using `csv.DictReader`:

# In[ ]:


# Note that we are creating a colon separated file
with open('colon_delimited_stock_prices.txt', 'w') as f:
    f.write("""date:symbol:closing_price
6/20/2020:AAPL:90.91
6/20/2020:MSFT:41.68
6/20/2020:FB:64.5
""")


# In[ ]:


with open('colon_delimited_stock_prices.txt') as f:
    colon_reader = csv.DictReader(f, delimiter=':')
    for dict_row in colon_reader:
        print(f"dict_row = {dict_row}")
        date = dict_row["date"]
        symbol = dict_row["symbol"]
        closing_price = float(dict_row["closing_price"])
        process(date, symbol, closing_price)


# Even if your file doesn’t have headers, you can still use DictReader by passing it the keys as a fieldnames parameter.

# In[ ]:


with open('tab_delimited_stock_prices.txt') as f:
    tab_dictreader = csv.DictReader(f, delimiter='\t', fieldnames = ["Date", "Symbol", "Closing_price"])
    for dict_row in tab_dictreader:
        print(f"dict_row = {dict_row}")
        date = dict_row["Date"]
        symbol = dict_row["Symbol"]
        closing_price = float(dict_row["Closing_price"])
        process(date, symbol, closing_price)


# You can similarly write out delimited data using `csv.writer`:

# In[ ]:


todays_prices = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5 }

# the newline="\n" part is necessary on Windows, newline="" would also work
with open('comma_delimited_stock_prices.txt', 'w', newline="\n") as f:
    csv_writer = csv.writer(f, delimiter=',')
    for stock, price in todays_prices.items():
        csv_writer.writerow([stock, price])

