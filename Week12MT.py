#!/usr/bin/env python
# coding: utf-8

# # Deep Learning

# *Deep learning* originally referred to the application of “deep” neural networks (that is,
# networks with more than one hidden layer), although in practice the term now
# encompasses a wide variety of neural architectures (including the “simple” neural
# networks we developed last week).
# 
# we’ll continue to build on our previous work and look at a wider variety of neural
# networks. To do so, we’ll introduce a number of abstractions that allow us to think
# about neural networks in a more general way.

# ## The Tensor

# Previously, we made a distinction between vectors (one-dimensional arrays) and
# matrices (two-dimensional arrays). When we start working with more complicated
# neural networks, we’ll need to use higher-dimensional arrays as well.
# 
# In many neural network libraries, n-dimensional arrays are referred to as tensors,
# which is what we’ll call them too.
# 
# Many packages do implement a full-featured
# `Tensor` class that overloads Python’s arithmetic operators and could handle a variety
# of other operations. Here we’ll cheat and say that a `Tensor` is just a `list`.
# - This is true in one direction—all of our vectors and matrices and higher-dimensional analogues are lists (of lists).
# - It is certainly not true in the other direction—most Python `lists` are not *n*-dimensional arrays in our sense.
# 
# Ideally we’d like to do something like:
# 
# ```python
# # A Tensor is either a float, or a List of Tensors
# Tensor = Union[float, List[Tensor]]
# ```
# 
# However, Python won’t let you define recursive types like that. And
# even if it did that definition is still not right, as it allows for bad
# “tensors” like:
# ```python
# [[1.0, 2.0],
#  [3.0]]
# ```
# whose rows have different sizes, which makes it not an *n*-dimensional
# array that we can work with.
# 
# So, as mentioned earlier, we’ll just cheat:

# In[ ]:


Tensor = list


# And we’ll write a helper function to find a tensor’s *shape*:

# In[ ]:


from typing import List

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes


# In[ ]:


print(shape([1, 2, 3])) # [3]
print(shape([[1, 2],
             [3, 4],
             [5, 6]])) # [3, 2]


# Because tensors can have any number of dimensions, we’ll typically need to work
# with them recursively. We’ll do one thing in the one-dimensional case and recurse in
# the higher-dimensional case:

# In[ ]:


def is_1d(tensor: Tensor) -> bool:
    """
    If tensor[0] is a list, it's a higher-order tensor.
    Otherwise, tensor is 1-dimensonal (that is, a vector).
    """
    return not isinstance(tensor[0], list)


# In[ ]:


print(is_1d([1, 2, 3]))
print(is_1d([[1, 2],
             [3, 4]]))


# which we can use to write a recursive *tensor_sum* function:

# In[ ]:


def tensor_sum(tensor: Tensor) -> float:
    """Sums up all the values in the tensor"""
    if is_1d(tensor):
        return sum(tensor)  # just a list of floats, use Python sum
    else:
        return sum(tensor_sum(tensor_i)      # Call tensor_sum on each row
                   for tensor_i in tensor)   # and sum up those results.


# In[ ]:


print(tensor_sum([1, 2, 3])) # 6
print(tensor_sum([[1, 2],
                  [3, 4]])) # 10


# If you’re not used to thinking recursively, you should ponder this until it makes sense,
# because we’ll use the same logic throughout this chapter. However, we’ll create a couple
# of helper functions so that we don’t have to rewrite this logic everywhere. The first
# applies a function elementwise to a single tensor:

# In[ ]:


from typing import Callable

# here f is a type of function that takes a single float as argument and returns a float
def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """Applies f elementwise"""
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]


# In[ ]:


print(tensor_apply(lambda x: x + 1, [1, 2, 3])) # [2, 3, 4]
print(tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]])) # [[2, 4], [6, 8]]


# We can use this to write a function that creates a zero tensor with the same shape as a
# given tensor:

# In[ ]:


def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)


# In[ ]:


print(zeros_like([1, 2, 3])) # [0, 0, 0]
print(zeros_like([[1, 2], [3, 4]])) # [[0, 0], [0, 0]]


# We’ll also need to apply a function to corresponding elements from two tensors
# (which had better be the exact same shape, although we won’t check that):

# In[ ]:


def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """Applies f to corresponding elements of t1 and t2"""
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i)
                for t1_i, t2_i in zip(t1, t2)]


# In[ ]:


import operator
print(tensor_combine(operator.add, [1, 2, 3], [4, 5, 6])) # [5, 7, 9]
print(tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6])) # [4, 10, 18]


# ## The Layer Abstraction

# Last week we built a simple neural net that allowed us to stack two layers
# of neurons, each of which computed `sigmoid(dot(weights, inputs))`.
# 
# Although that’s perhaps an idealized representation of what an actual neuron does, in
# practice we’d like to allow a wider variety of things.
# - Perhaps we’d like the neurons to remember something about their previous inputs. 
# 
# - Perhaps we’d like to use a different activation function than sigmoid. 
# 
# - And frequently we’d like to use more than two layers. (Our `feed_forward` function actually handled any number of layers, but our gradient computations did not.)
# 
# In this chapter we’ll build machinery for implementing such a variety of neural networks.
# Our fundamental abstraction will be the `Layer`, something that knows how to
# apply some function to its inputs and that knows how to backpropagate gradients.
# 
# One way of thinking about the neural networks we built last week is as a “linear”
# layer (performing dot product between inputs and weights), followed by a “sigmoid” layer (apply sigmoid function to the sum resulted from the dot product), then another linear layer and another sigmoid
# layer. We didn’t distinguish them in these terms, but doing so will allow us to experiment
# with much more general structures:

# In[ ]:


from typing import Iterable, Tuple

class Layer:
    """
    Our neural networks will be composed of Layers, each of which
    knows how to do some computation on its inputs in the "forward"
    direction and propagate gradients in the "backward" direction.
    """
    def forward(self, input):
        """
        Note the lack of types. We're not going to be prescriptive
        about what kinds of inputs layers can take and what kinds
        of outputs they can return.
        """
        raise NotImplementedError

    def backward(self, gradient):
        """
        Similarly, we're not going to be prescriptive about what the
        gradient looks like. It's up to you the user to make sure
        that you're doing things sensibly.
        """
        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        """
        Returns the parameters of this layer. The default implementation
        returns nothing, so that if you have a layer with no parameters
        you don't have to implement this.
        """
        return ()

    def grads(self) -> Iterable[Tensor]:
        """
        Returns the gradients, in the same order as params()
        """
        return ()


# The `forward` and `backward` methods will have to be implemented in our concrete
# subclasses. Once we build a neural net, we’ll want to train it using gradient descent,
# which means we’ll want to update each parameter in the network using its gradient.
# Accordingly, we insist that each layer be able to tell us its parameters and gradients.
# 
# Some layers (for example, a layer that applies *sigmoid* to each of its inputs) have no
# parameters to update, so we provide a default implementation that handles that case.
# 
# Let’s look at that layer:

# In[ ]:


from Week11M import sigmoid


# In[ ]:


class Sigmoid(Layer):
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply sigmoid to each element of the input tensor,
        and save the results to use in backpropagation.
        """
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad,
                              self.sigmoids,
                              gradient)


# There are a couple of things to notice here. One is that during the forward pass we
# saved the computed sigmoids so that we could use them later in the backward pass.
# Our layers will typically need to do this sort of thing.
# 
# Second, you may be wondering where the `sig * (1 - sig) * grad` comes from.
# This is the chain rule from calculus and corresponds to the `output * (1 - output) * (output - target)` term in our previous neural networks.
# 
# Finally, you can see how we were able to make use of the `tensor_apply` and the `tensor_combine` functions. Most of our layers will use these functions similarly.

# ## The Linear Layer

# The other piece we’ll need to duplicate the neural networks from last week is a “linear”
# layer that represents the `dot(weights, inputs)` part of the neurons.
# 
# This layer will have parameters, which we’d like to initialize with random values.
# 
# It turns out that the initial parameter values can make a huge difference in how
# quickly (and sometimes *whether*) the network trains. If weights are too big, they may
# produce large outputs in a range where the activation function has near-zero gradients.
# And parts of the network that have zero gradients necessarily can’t learn anything
# via gradient descent.
# 
# Accordingly, we’ll implement three different schemes for randomly generating our
# weight tensors.
# 1. The first is to choose each value from the random uniform distribution on [0, 1]—that is, as a `random.random()`.
# 2. The second (and default) is to choose each value randomly from a standard normal distribution.
# 3. And the third is to use *Xavier initialization*, where each weight is initialized with a random draw from a normal distribution with mean 0 and variance `2 / (num_inputs + num_outputs)`. It turns out this often works nicely for neural network weights. We’ll implement these with a `random_uniform` function and a `random_normal` function:

# In[ ]:


import random

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]


# In[ ]:


from scipy.stats import norm

def random_normal(*dims: int,
                  mean: float = 0.0,
                  variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * norm.ppf(random.random())
                for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance)
                for _ in range(dims[0])]


# In[ ]:


ran1 = random_uniform(2, 3, 4)
print(ran1)
print(shape(ran1))

ran2 = random_normal(5, 6, mean=10)
print(ran2)
print(shape(ran2))


# And then wrap them all in a `random_tensor` function:

# In[ ]:


def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")


# Now we can define our linear layer. We need to initialize it with the dimension of the
# inputs (which tells us how many weights each neuron needs), the dimension of the
# outputs (which tells us how many neurons we should have), and the initialization
# scheme we want:

# In[ ]:


from numpy import dot

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, init: str = 'xavier') -> None:
        """
        A layer of output_dim neurons, each with input_dim weights
        (and a bias).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.w[o] is the weights for the o-th neuron
        self.w = random_tensor(output_dim, input_dim, init=init)

        # self.b[o] is the bias term for the o-th neuron
        self.b = random_tensor(output_dim, init=init)

    # The forward method is easy to implement. We’ll get one output per neuron, which
    # we stick in a vector. And each neuron’s output is just the dot of its weights with the
    # input, plus its bias:
    def forward(self, input: Tensor) -> Tensor:
        # Save the input to use in the backward pass.
        self.input = input

        # Return the vector of neuron outputs.
        return [dot(input, self.w[o]) + self.b[o]
                for o in range(self.output_dim)]

    # The backward method is more involved and requires calculus
    # Here the input parameter 'gradient' corresponds to the delta value in our earlier (optional) derivation
    # of the backpropogation algorithm.
    def backward(self, gradient: Tensor) -> Tensor:
        
        # computes the rate of change of the loss function with respect to the current layer's weights and biases
        # for bias, the rate of change happens to be just delta
        self.b_grad = gradient 
        # for weights, the rate of change is delta * input of the current layer
        # refer to 'delta_2 * h' and 'delta_1 * x' in our (optional) backpropagation derivation 
        self.w_grad = [[self.input[i] * gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]
        
        # computes the new delta to be backpropagated into previous layers
        # refer to 'delta_2 * w2' in our (optional) backpropagation derivation
        return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]


# ## Neural Networks as a Sequence of Layers

# We’d like to think of neural networks as sequences of layers, so let’s come up with a
# way to combine multiple layers into one. The resulting neural network is itself a layer,
# and it implements the `Layer` methods in the obvious ways:

# In[ ]:


from typing import List

class Sequential(Layer):
    """
    A layer consisting of a sequence of other layers.
    It's up to you to make sure that the output of each layer
    makes sense as the input to the next layer.
    """
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, input):
        """Just forward the input through the layers in order."""
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient):
        """Just backpropagate the gradient through the layers in reverse."""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> Iterable[Tensor]:
        """Just return the params from each layer."""
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        """Just return the grads from each layer."""
        return (grad for layer in self.layers for grad in layer.grads())


# So we could represent the neural network we used for XOR as:
# 
# ```python
# xor_net = Sequential([
#     Linear(input_dim=2, output_dim=2),
#     Sigmoid(),
#     Linear(input_dim=2, output_dim=1),
#     Sigmoid()
# ])
# ```
# But we still need a little more machinery to train it.
