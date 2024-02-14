#!/usr/bin/env python
# coding: utf-8

# ## Scraping the Web

# Another way to get data is by scraping it from web pages. Fetching web pages, it turns out, is pretty easy; getting meaningful structured information out of them less so.
# 
# Pages on the web are written in HTML, in which text is (ideally) marked up into elements and their attributes:

# ```
# <html>
#     <head>
#     <title>A web page</title>
#     </head>
#     <body>
#         <p id="author">Joel Grus</p>
#         <p id="subject">Data Science</p>
#     </body>
# </html>
# ```

# In a perfect world, where all web pages were marked up semantically for our benefit, we would be able to extract data using rules like “find the `<p>` element whose id is subject and return the text it contains.”
# 
# In the actual world, HTML is not generally well formed, let alone annotated. This means we’ll need help making sense of it.
# 
# We will be using a couple of packages to get data out of HTML.
# - Beautiful Soup library, which builds a tree out of the various elements on a web page and provides a simple interface for accessing them
# - Requests library, which is a much nicer way of making HTTP requests than anything that’s built into Python
# - html5lib library, which is able to handle HTML that's not perfectly formed better than Python's built-in HTML parser
# 
# If you installed Anaconda, these libraries should have already been installed. Otherwise, you may need to install them yourself.

# In[ ]:


# From Jupyter Notebook, run
get_ipython().system('pip install beautifulsoup4 requests html5lib')


# From console/shell/prompt, run
# 
# `python -m pip install beautifulsoup4 requests html5lib`

# To use Beautiful Soup, we pass a string containing HTML into the BeautifulSoup function. In our examples, this will be the result of a call to `requests.get`:

# In[ ]:


from bs4 import BeautifulSoup
import requests

url = ("https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html")
html = requests.get(url).text
print(html)
soup = BeautifulSoup(html, 'html5lib')


# We’ll typically work with `Tag` objects, which correspond to the tags representing the structure of an HTML page.
# 
# For example, to find the first `<p>` tag (and its contents), you can use:

# In[ ]:


first_paragraph = soup.find('p')        # or just soup.p
print(first_paragraph)


# You can get the text contents of a `Tag` using its `text` property:

# In[ ]:


first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

print(first_paragraph_text)
print(first_paragraph_words)


# And you can extract a tag’s attributes by treating it like a `dict`:

# In[ ]:


first_paragraph_id = soup.p['id']       # raises KeyError if no 'id'
first_paragraph_id2 = soup.p.get('id')  # returns None if no 'id'

print(first_paragraph_id)
print(first_paragraph_id2)


# You can get multiple tags at once as follows:

# In[ ]:


all_paragraphs = soup.find_all('p')  # or just soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

print(all_paragraphs)
print(paragraphs_with_ids)


# Frequently, you’ll want to find tags with a specific class:

# In[ ]:


important_paragraphs = soup('p', {'class' : 'important'}) 
important_paragraphs2 = soup('p', 'important')

print(important_paragraphs)
print(important_paragraphs2)


# And you can combine these methods to implement more elaborate logic. For example, if you want to find every `<span>` element that is contained inside a `<div>` element, you could do this:

# In[ ]:


spans_inside_divs = [span
                     for div in soup('div')     # for each <div> on the page
                     for span in div('span')]   # find each <span> inside it

print(spans_inside_divs)


# Of course, the important data won’t typically be labeled as class="important". You’ll need to carefully inspect the source HTML, reason through your selection logic, and worry about edge cases to make sure your data is correct.

# ## Example: Keeping Tabs on Congress

# The VP of Policy at your start-up company is worried about potential regulation of the data science industry and asks you to quantify what Congress is saying on the topic. In particular, he wants you to find all the representatives who have press releases about "data."
# 
# There is a page with links to all of the representatives' websites at https://www.house.gov/representatives
# 
# And if you "view source," all of the links to the websites look like:
# ```
# <td>
#     <a href="https://jayapal.house.gov">Jayapal, Pramila</a>
# </td>
# ```
# Let’s start by collecting all of the URLs linked to from that page:

# In[ ]:


from bs4 import BeautifulSoup
import requests

url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

all_urls = [a['href']
            for a in soup('a')     # Note: the <a> tag is used to define a hyperlink
            if a.has_attr('href')]
print(len(all_urls))


# This returns way too many URLs. If you look at them, the ones we want start with either `http://` or `https://`, have some kind of name, and end with either `.house.gov` or `.house.gov/`.
# 
# This is a good place to use a regular expression:

# In[ ]:


import re

# Must start with http:// or https://
# Must end with .house.gov or .house.gov/
# For references on generating regular expressions, see
# https://docs.python.org/3/library/re.html
# https://www.w3schools.com/python/python_regex.asp
regex = r"^https?://.*\.house\.gov/?$"

# Let's write some tests!
assert re.match(regex, "http://joel.house.gov")
assert re.match(regex, "https://joel.house.gov")
assert re.match(regex, "http://joel.house.gov/")
assert re.match(regex, "https://joel.house.gov/")
assert not re.match(regex, "joel.house.gov")
assert not re.match(regex, "http://joel.house.com")
assert not re.match(regex, "https://joel.house.gov/biography")

# And now apply
good_urls = [url for url in all_urls if re.match(regex, url)]
print(len(good_urls))


# That’s still way too many, as there are only 435 representatives. If you look at the list, there are a lot of duplicates. Let’s use set to get rid of them:

# In[ ]:


good_urls = list(set(good_urls))
print(len(good_urls))


# So the number did not turn out to be exactly 435. Maybe someone has more than one website. In any case, this is good enough.
# 
# When we look at the sites, most of them have a link to press releases. For example:

# In[ ]:


html = requests.get('https://susielee.house.gov/').text
soup = BeautifulSoup(html, 'html5lib')

# Use a set because the links might appear multiple times.
links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
print(links) # {'/media/press-releases'}


# Notice that this is a relative link, which means we need to remember the originating site. Let’s do some scraping:

# In[ ]:


from typing import Dict, Set

press_releases: Dict[str, Set[str]] = {}
    
for house_url in good_urls:
    html = requests.get(house_url).text
    soup = BeautifulSoup(html, 'html5lib')
    pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
    
    print(f"{house_url}: {pr_links}")
    press_releases[house_url] = pr_links


# Note: Normally it is impolite to scrape a site freely like this. Most sites will have a robots.txt file that indicates how frequently you may scrape the site (and which paths you’re not supposed to scrape), but since it’s Congress we don’t need to be particularly polite.
# 
# If you watch these as they scroll by, you’ll see a lot of */media/press-releases* and *media-center/press-releases*, as well as various other addresses. One of these URLs is https://susielee.house.gov/media/press-releases.
# 
# Remember that our goal is to find out which congresspeople have press releases mentioning "data." We’ll write a slightly more general function that checks whether a page of press releases mentions any given term.
# 
# If you visit the site and view the source, it seems like there’s a snippet from each press release inside a `<p>` tag, so we’ll use that as our first attempt:

# In[ ]:


def paragraph_mentions(text: str, keyword: str) -> bool:
    """
    Returns True if a <p> inside the text mentions {keyword}
    """
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]

    return any(keyword.lower() in paragraph.lower() for paragraph in paragraphs)


# Let’s write a quick test for it:

# In[ ]:


text = """<body><h1>Facebook</h1><p>Twitter</p>"""
assert paragraph_mentions(text, "twitter")       # is inside a <p>
assert not paragraph_mentions(text, "facebook")  # not inside a <p>


# At last we’re ready to find the relevant congresspeople and give their names to the VP:

# In[ ]:


for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = f"{house_url}/{pr_link}"
        text = requests.get(url).text

        if paragraph_mentions(text, 'data'):
            print(f"{house_url}")
            break # done with this house_url


# Note: If you look at the various “press releases” pages, most of them are paginated with only 5 or 10 press releases per page. This means that we only retrieved the few most recent press releases for each congressperson. A more thorough solution would have iterated over the pages and retrieved the full text of each press release.
