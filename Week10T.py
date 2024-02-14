#!/usr/bin/env python
# coding: utf-8

# # Clustering (Continued)

# ## Bottom-Up Hierarchical Clustering

# An alternative approach to clustering is to “grow” clusters from the bottom up. We
# can do this in the following way:
# 1. Make each input its own cluster of one.
# 2. As long as there are multiple clusters remaining, find the two closest clusters and merge them.
# 
# At the end, we’ll have one giant cluster containing all the inputs. If we keep track of
# the merge order, we can re-create any number of clusters by unmerging.
# - For example, if we want three clusters, we can just undo the last two merges.
# 
# We’ll use a really simple representation of clusters. Our values will live in *leaf* clusters,
# which we will represent as `NamedTuples`:

# In[ ]:


from typing import List

inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]


# In[ ]:


from typing import NamedTuple, Union
Vector = List[float]

class Leaf(NamedTuple):
    value: Vector

leaf1 = Leaf([10,  20])
leaf2 = Leaf([30, -15])


# We’ll use these to grow merged clusters, which we will also represent as `NamedTuples`:

# In[ ]:


class Merged(NamedTuple):
    children: tuple
    order: int

merged = Merged((leaf1, leaf2), order=1)

Cluster = Union[Leaf, Merged]


# We’ll talk about merge order in a bit, but first let’s create a helper function that recursively
# returns all the values contained in a (possibly merged) cluster:

# In[ ]:


def get_values(cluster: Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value
                for child in cluster.children
                for value in get_values(child)]


# In[ ]:


print(get_values(merged))


# In order to merge the closest clusters, we need some notion of the distance between
# clusters.
# - We’ll use the *minimum* distance between elements of the two clusters, which merges the two clusters that are closest to touching (but will sometimes produce large chain-like clusters that aren’t very tight).
# - If we wanted tight spherical clusters, we might use the *maximum* distance instead, as it merges the two clusters that fit in the smallest ball.
# - Both are common choices, as is the *average* distance:

# In[ ]:


from typing import Callable
from scipy.spatial import distance

def cluster_distance(cluster1: Cluster,
                     cluster2: Cluster,
                     distance_agg: Callable = min) -> float:
    """
    compute all the pairwise distances between cluster1 and cluster2
    and apply the aggregation function _distance_agg_ to the resulting list
    """
    return distance_agg([distance.euclidean(v1, v2)
                         for v1 in get_values(cluster1)
                         for v2 in get_values(cluster2)])


# We’ll use the merge order slot to track the order in which we did the merging. Smaller
# numbers will represent *later* merges. This means when we want to unmerge clusters, we do so from lowest merge order to highest. Since `Leaf` clusters were never merged,
# we’ll assign them infinity, the highest possible value. And since they don’t have
# an `.order` property, we’ll create a helper function:

# In[ ]:


def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float('inf')  # was never merged
    else:
        return cluster.order


# Similarly, since `Leaf` clusters don’t have children, we’ll create and add a helper function
# for that:

# In[ ]:


from typing import Tuple

def get_children(cluster: Cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children


# Now we’re ready to create the clustering algorithm:

# In[ ]:


def bottom_up_cluster(inputs: List[Vector],
                      distance_agg: Callable = min) -> Cluster:
    # Start with all leaves
    clusters: List[Cluster] = [Leaf(input) for input in inputs]

    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
        return cluster_distance(pair[0], pair[1], distance_agg)

    # as long as we have more than one cluster left...
    while len(clusters) > 1:
        # find the two closest clusters
        c1, c2 = min(((cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]),
                      key=pair_distance)

        # remove them from the list of clusters
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge them, using merge_order = # of clusters left
        merged_cluster = Merged((c1, c2), order=len(clusters))

        # and add their merge
        clusters.append(merged_cluster)

    # when there's only one cluster left, return it
    return clusters[0]


# Its use is very simple:

# In[ ]:


base_cluster = bottom_up_cluster(inputs)


# This produces a clustering that looks as follows:
# ![hierarchy_clusters](hierarchy_clusters.png)
# 
# The numbers at the top indicate “merge order.” Since we had 20 inputs, it took 19
# merges to get to this one cluster. The first merge created cluster 18 by combining the
# leaves [19, 28] and [21, 27]. And the last merge created cluster 0.
# 
# If you wanted only two clusters, you’d split at the first fork (“0”), one branch moving down and the other moving right, creating one cluster
# with six points and a second with the rest. For three clusters, you’d continue to the
# second fork (“1”), which indicates to split that first cluster into the cluster with ([19,
# 28], [21, 27], [20, 23], [26, 13]) and the cluster with ([11, 15], [13, 13]). And so on.
# 
# Generally, though, we don’t want to be squinting at nasty text representations like
# this. Instead, let’s write a function that generates any number of clusters by performing
# the appropriate number of unmerges:

# In[ ]:


def generate_clusters(base_cluster: Cluster,
                      num_clusters: int) -> List[Cluster]:
    # start with a list with just the base cluster
    clusters = [base_cluster]

    # as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
        # choose the last-merged of our clusters
        next_cluster = min(clusters, key=get_merge_order)
        # remove it from the list
        clusters = [c for c in clusters if c != next_cluster]

        # and add its children to the list (i.e., unmerge it)
        clusters.extend(get_children(next_cluster)) # extends will unpack the children and add each of them to the list

    # once we have enough clusters...
    return clusters


# So, for example, if we want to generate three clusters, we can just do:

# In[ ]:


three_clusters = [get_values(cluster)
                  for cluster in generate_clusters(base_cluster, 3)]


# which we can easily plot:

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

for i, cluster, marker, color in zip([1, 2, 3],
                                     three_clusters,
                                     ['D','o','*'],
                                     ['r','g','b']):
    xs, ys = zip(*cluster)  # magic unzipping trick
    plt.scatter(xs, ys, color=color, marker=marker)

    # put a number at the mean of the cluster
    x, y = np.mean(cluster, axis = 0)
    plt.plot(x, y, marker='$' + str(i) + '$', color='black')

plt.title("User Locations -- 3 Bottom-Up Clusters, Min")
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.show()


# This gives very different results than *k*-means did.

# As mentioned previously, this is because using `min` in `cluster_distance` tends to
# give chain-like clusters. If we instead use `max` (which gives tight clusters), it looks the
# same as the 3-means result.

# In[ ]:


base_cluster2 = bottom_up_cluster(inputs, max)

three_clusters2 = [get_values(cluster)
                  for cluster in generate_clusters(base_cluster2, 3)]

for i, cluster, marker, color in zip([1, 2, 3],
                                     three_clusters2,
                                     ['D','o','*'],
                                     ['r','g','b']):
    xs, ys = zip(*cluster)  # magic unzipping trick
    plt.scatter(xs, ys, color=color, marker=marker)

    # put a number at the mean of the cluster
    x, y = np.mean(cluster, axis = 0)
    plt.plot(x, y, marker='$' + str(i) + '$', color='black')

plt.title("User Locations -- 3 Bottom-Up Clusters, Min")
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.show()


# Note: The previous `bottom_up_clustering` implementation is relatively
# simple, but also shockingly inefficient. In particular, it recomputes
# the distance between each pair of inputs at every step. A more efficient
# implementation might instead precompute the distances
# between each pair of inputs and then perform a lookup inside `cluster_distance`. A really efficient implementation would likely also
# remember the `cluster_distances` from the previous step.

# ## K-Means in scikit-learn

# In[ ]:


# adapted from https://machinelearningmastery.com/clustering-algorithms-with-python/

# k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# generate an artificial dataset
# when dealing with real datasets, you probably want to load the data via pandas
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

print(f"X = {X}")

# define the model
model = KMeans(n_clusters=2)

# fit the model
model.fit(X)

# assign a cluster to each example
yhat = model.predict(X)

print(f"yhat = {yhat}")


# retrieve unique clusters
clusters = unique(yhat)
print(f"clusters = {clusters}")


# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    
    print(f"row_ix = {row_ix}")
    
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
    
# show the plot
plt.show()


# You can learn more about other types of clustering in:
# 
# https://scikit-learn.org/stable/modules/clustering.html
# 
# https://machinelearningmastery.com/clustering-algorithms-with-python/
# 
