#!/usr/bin/env python
# coding: utf-8

# # Deep Learning (Continued)

# In[1]:


from Week12MT import *


# ## Loss and Optimization

# Previously we wrote out individual loss functions and gradient functions for our
# models. Here we’ll want to experiment with different loss functions, so (as usual) we’ll
# introduce a new `Loss` abstraction that encapsulates both the loss computation and the
# gradient computation:

# In[2]:


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """How good are our predictions? (Larger numbers are worse.)"""
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """How does the loss change as the predictions change?"""
        raise NotImplementedError


# We’ve already worked many times with the loss that’s the sum of the squared errors,
# so we should have an easy time implementing that. The only trick is that we’ll need to
# use `tensor_combine`:

# In[3]:


class SSE(Loss):
    """Loss function that computes the sum of the squared errors."""
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # Compute the tensor of squared differences
        squared_errors = tensor_combine(
            lambda predicted, actual: (predicted - actual) ** 2,
            predicted,
            actual)

        # And just add them up
        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted, actual: 2 * (predicted - actual),
            predicted,
            actual)


# In[4]:


sse_loss = SSE()
assert sse_loss.loss([1, 2, 3], [10, 20, 30]) == 9 ** 2 + 18 ** 2 + 27 ** 2
assert sse_loss.gradient([1, 2, 3], [10, 20, 30]) == [-18, -36, -54]


# (We’ll look at a different loss function in a bit.)
# 
# The last piece to figure out is gradient descent. We'll introduce a `Optimizer` abstraction, of which gradient
# descent will be a specific instance:
# 

# In[5]:


class Optimizer:
    """
    An optimizer updates the weights of a layer (in place) using information
    known by either the layer or the optimizer (or by both).
    """
    def step(self, layer: Layer) -> None:
        raise NotImplementedError


# After that it’s easy to implement gradient descent, again using `tensor_combine`:

# In[11]:


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # Update param using a gradient step
            param[:] = tensor_combine(
                lambda param, grad: param - grad * self.lr,
                param,
                grad)


# The only thing that’s maybe surprising is the “slice assignment,” which is a reflection
# of the fact that reassigning a list doesn’t change its original value. That is, if you just
# did `param = tensor_combine(. . .)`, you would be redefining the local variable `param`, but you would not be affecting the original parameter tensor stored in the
# layer. If you assign to the slice `[:]`, however, it actually changes the values inside the
# list.
# 
# Here’s a simple example to demonstrate:

# In[6]:


tensor = [[1, 2], [3, 4]]

for row in tensor:
    row = [0, 0]
assert tensor == [[1, 2], [3, 4]], "assignment doesn't update a list"

for row in tensor:
    row[:] = [0, 0]
assert tensor == [[0, 0], [0, 0]], "but slice assignment does"


# To demonstrate the value of this abstraction, let’s implement another optimizer that
# uses *momentum*. The idea is that we don’t want to overreact to each new gradient, and
# so we maintain a running average of the gradients we’ve seen, updating it with each
# new gradient and taking a step in the direction of the average:

# In[7]:


class Momentum(Optimizer):
    def __init__(self,
                 learning_rate: float,
                 momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []  # running average

    def step(self, layer: Layer) -> None:
        # If we have no previous updates, start with all zeros.
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates,
                                       layer.params(),
                                       layer.grads()):
            # Apply momentum
            update[:] = tensor_combine(
                lambda u, g: self.mo * u + (1 - self.mo) * g,
                update,
                grad)

            # Then take a gradient step
            param[:] = tensor_combine(
                lambda p, u: p - self.lr * u,
                param,
                update)


# Because we used an `Optimizer` abstraction, we can easily switch between our different
# optimizers.

# ## Example: XOR Revisited

# Let’s see how easy it is to use our new framework to train a network that can compute
# XOR. We start by re-creating the training data:

# In[10]:


# training data
xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]


# and then we define the network, although now we can leave off the last sigmoid layer:

# In[14]:


random.seed(0)

net = Sequential([
      Linear(input_dim=2, output_dim=2),
      Sigmoid(),
      Linear(input_dim=2, output_dim=1)
])


# We can now write a simple training loop, except that now we can use the abstractions
# of `Optimizer` and `Loss`. This allows us to easily try different ones:

# In[15]:


import tqdm

optimizer = GradientDescent(learning_rate=0.1)
loss = SSE()

with tqdm.trange(3000) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

        t.set_description(f"xor loss {epoch_loss:.3f}")


# This should train quickly, and you should see the loss go down. And now we can
# inspect the weights:

# In[16]:


for param in net.params():
    print(param)


# In[17]:


for x in xs:
    print(net.forward(x))


# Notice that this network learned different features than the one we trained last week, but it still manages to do the same thing.

# ## Other Activation Functions

# The sigmoid function has fallen out of favor for a couple of reasons.
# - One reason is that `sigmoid(0)` equals 1/2, which means that a neuron whose inputs sum to 0 has a positive output.
# - Another is that its gradient is very close to 0 for very large and very small inputs, which means that its gradients can get “saturated” and its weights can get stuck.
# 
# One popular replacement is `tanh` (“hyperbolic tangent”), which is a different
# sigmoid-shaped function that ranges from –1 to 1 and outputs 0 if its input is 0. The
# derivative of `tanh(x)` is just `1 - tanh(x) ** 2`, which makes the layer easy to write:

# In[20]:


import math

def tanh(x: float) -> float:
    # If x is very large or very small, tanh is (essentially) 1 or -1.
    # We check for this because e.g. math.exp(1000) raises an error.
    if x < -100:  return -1
    elif x > 100: return 1

    em2x = math.exp(-2 * x)
    return (1 - em2x) / (1 + em2x)

class Tanh(Layer):
    def forward(self, input: Tensor) -> Tensor:
        # Save tanh output to use in backward pass.
        self.tanh = tensor_apply(tanh, input)
        return self.tanh

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(
            lambda tanh, grad: (1 - tanh ** 2) * grad,
            self.tanh,
            gradient)


# In larger networks another popular replacement is `Relu`, which is 0 for negative
# inputs and the identity for positive inputs:

# In[19]:


class Relu(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return tensor_apply(lambda x: max(x, 0), input)

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda x, grad: grad if x > 0 else 0,
                              self.input,
                              gradient)


# There are many others. I encourage you to play around with them in your networks.

# ## Example: FizzBuzz Revisited

# We can now use our “deep learning” framework to reproduce our solution from
# “Fizz Buzz” example. Let’s set up the data:

# In[21]:


Vector = List[float]

def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def binary_encode(x: int) -> Vector:
    binary: List[float] = []

    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary

def argmax(xs: list) -> int:
    """Returns the index of the largest value"""
    return max(range(len(xs)), key=lambda i: xs[i])

xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]


# and create the network:

# In[22]:


NUM_HIDDEN = 25

random.seed(0)

net = Sequential([
    Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform'),
    Sigmoid()
])


# As we’re training, let’s also track our accuracy on the training set:

# In[23]:


def fizzbuzz_accuracy(low: int, hi: int, net: Layer) -> float:
    num_correct = 0
    for n in range(low, hi):
        x = binary_encode(n)
        predicted = argmax(net.forward(x))
        actual = argmax(fizz_buzz_encode(n))
        if predicted == actual:
            num_correct += 1

    return num_correct / (hi - low)


# In[24]:


optimizer = Momentum(learning_rate=0.1, momentum=0.9)
loss = SSE()

with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

        accuracy = fizzbuzz_accuracy(101, 1024, net)
        t.set_description(f"fb loss: {epoch_loss:.2f} acc: {accuracy:.2f}")

# Now check results on the test set
print("test results", fizzbuzz_accuracy(1, 101, net))


# After 500 training iterations, the model gets close to 90% accuracy on the test set; if you
# keep training it longer, it should do even better. (I don’t think it’s possible to train to
# 100% accuracy with only 25 hidden units, but it should be possible if you go up to 50
# hidden units.)

# ## Softmaxes and Cross-Entropy

# The neural net we used in the previous section ended in a `Sigmoid` layer, which
# means that its output was a vector of numbers between 0 and 1. In particular, it could
# output a vector that was entirely 0s, or it could output a vector that was entirely 1s.
# 
# Yet when we’re doing classification problems, we’d like to output a 1 for the correct
# class and a 0 for all the incorrect classes. Generally our predictions will not be so perfect,
# but we’d at least like to predict an actual probability distribution over the classes.
# - For example, if we have two classes, and our model outputs [0, 0], it’s hard to make much sense of that. It doesn’t think the output belongs in either class?
# - But if our model outputs [0.4, 0.6], we can interpret it as a prediction that there’s a probability of 0.4 that our input belongs to the first class and 0.6 that our input belongs to the second class.
# 
# In order to accomplish this, we typically forgo the final `Sigmoid` layer and instead use
# the `softmax` function, which converts a vector of real numbers to a vector of probabilities.
# 
# We compute `exp(x)` for each number in the vector, which results in a vector
# of positive numbers. After that, we just divide each of those positive numbers by the
# sum, which gives us a bunch of positive numbers that add up to 1—that is, a vector of
# probabilities.
# 
# If we ever end up trying to compute, say, `exp(1000)` we will get a Python error, so
# before taking the exp we subtract off the largest value. This turns out to result in the
# same probabilities; it’s just safer to compute in Python:

# In[25]:


def softmax(tensor: Tensor) -> Tensor:
    """Softmax along the last dimension"""
    if is_1d(tensor):
        # Subtract largest value for numerical stabilitity.
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]

        sum_of_exps = sum(exps)                 # This is the total "weight".
        return [exp_i / sum_of_exps             # Probability is the fraction
                for exp_i in exps]              # of the total weight.
    else:
        return [softmax(tensor_i) for tensor_i in tensor]


# Once our network produces probabilities, we often use a different loss function called
# cross-entropy (or sometimes “negative log likelihood”).
# 
# Typically we won’t include the `softmax` function as part of the neural network itself.
# This is because it turns out that if `softmax` is part of your loss function but not part of
# the network itself, the gradients of the loss with respect to the network outputs are
# very easy to compute.

# In[26]:


class SoftmaxCrossEntropy(Loss):
    """
    This is the negative-log-likelihood of the observed values, given the
    neural net model. So if we choose weights to minimize it, our model will
    be maximizing the likelihood of the observed data.
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # Apply softmax to get probabilities
        probabilities = softmax(predicted)

        # This will be log p_i for the actual class i and 0 for the other
        # classes. We add a tiny amount to p to avoid taking log(0).
        likelihoods = tensor_combine(lambda p, act: math.log(p + 1e-30) * act,
                                     probabilities,
                                     actual)

        # And then we just sum up the negatives.
        return -tensor_sum(likelihoods)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)

        # Isn't this a pleasant equation?
        return tensor_combine(lambda p, actual: p - actual,
                              probabilities,
                              actual)


# If I now train the same Fizz Buzz network using SoftmaxCrossEntropy loss, I find
# that it typically trains much faster (that is, in many fewer epochs).
# Presumably this may be explained as follows.
# 
# If I need to predict class 0 (a vector with a 1 in the first position and 0s in the
# remaining positions), in the `linear + sigmoid` case I need the first output to be a
# large positive number and the remaining outputs to be large negative numbers. In the
# softmax case, however, I just need the first output to be larger than the remaining
# outputs. Clearly there are a lot more ways for the second case to happen, which suggests
# that it should be easier to find weights that make it so:

# In[27]:


random.seed(0)

net = Sequential([
    Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform')
    # No final sigmoid layer now
])

optimizer = Momentum(learning_rate=0.1, momentum=0.9)
loss = SoftmaxCrossEntropy()

with tqdm.trange(100) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

        accuracy = fizzbuzz_accuracy(101, 1024, net)
        t.set_description(f"fb loss: {epoch_loss:.3f} acc: {accuracy:.2f}")

# Again check results on the test set
print("test results", fizzbuzz_accuracy(1, 101, net))

