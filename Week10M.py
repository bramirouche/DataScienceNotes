#!/usr/bin/env python
# coding: utf-8

# # Clustering (Continued)

# In[ ]:


from Week09R import *


# ## Example: Clustering Colors

# The VP of Swag has designed attractive stickers that he’d like you to
# hand out at meetups. Unfortunately, your sticker printer can print at most five colors
# per sticker. And since the VP of Art is on sabbatical, the VP of Swag asks if there’s
# some way you can take his design and modify it so that it contains only five colors.
# 
# Computer images can be represented as two-dimensional arrays of pixels, where each
# pixel is itself a three-dimensional vector `(red, green, blue)` indicating its color (note: sometimes there's a fourth dimension used to indicate the level of transparency, but we will not deal with it here).
# 
# Creating a five-color version of the image, then, entails:
# 1. Choosing five colors.
# 2. Assigning one of those colors to each pixel.
# 
# It turns out this is a great task for k-means clustering, which can partition the pixels
# into five clusters in red-green-blue space. If we then recolor the pixels in each cluster
# to the mean color, we’re done.
# 
# To start with, we’ll need a way to load an image into Python. We can do this with matplotlib,
# if we first install the `pillow` library:

# In[ ]:


# pillow is pre-installed with Anaconda, so only manually install it if you did not install anaconda 
get_ipython().system('python -m pip install pillow')


# Then we can just use `matplotlib.image.imread`:

# In[ ]:


image_path = r"Uga_X.jpg"             # or wherever your image is
import matplotlib.image as mpimg
img = mpimg.imread(image_path) / 256  # rescale to between 0 and 1


# The image we just loaded is one of Uga X. Of course, you could try other images as well.
# 
# ![Uga_X](Uga_X.jpg)

# Behind the scenes `img` is a NumPy array, but for our purposes, we can treat it as a 3-dimensional list.

# In[ ]:


print(type(img))
print(img.shape)


# `img[i][j]` is the pixel in the *i*th row and *j*th column, and each pixel is a list `[red, green, blue]` of numbers between 0 and 1 indicating the color of that pixel:

# In[ ]:


top_row = img[0]
top_left_pixel = top_row[0]
red, green, blue = top_left_pixel

print(f"top row =\n{top_row}\n")
print(f"top_left_pixel = {top_left_pixel}")
print(f"red, green, blue = {red}, {green}, {blue}")


# In particular, we can get a flattened list of all the pixels as:

# In[ ]:


# .tolist() converts a numpy array to a Python list
pixels = [pixel.tolist() for row in img for pixel in row]

print(type(pixels))
print(len(pixels)) # 55000 = 275 * 200
print(pixels[0])


# and then feed them to our clusterer:

# In[ ]:


clusterer = KMeans(5)
clusterer.train(pixels)   # this might take a while


# Once it finishes, we just construct a new image with the same format:

# In[ ]:


def recolor(pixel: Vector) -> Vector:
    cluster = clusterer.classify(pixel)        # index of the closest cluster
    return clusterer.means[cluster]            # mean of the closest cluster


# In[ ]:


new_img = [[recolor(pixel) for pixel in row]   # recolor this row of pixels
           for row in img]                     # for each row in the image


# and display it, using `plt.imshow`:

# In[ ]:


plt.rcParams['figure.figsize'] = (5, 5)

plt.imshow(new_img)
plt.axis('off')
plt.show()

