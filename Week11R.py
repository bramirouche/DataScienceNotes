#!/usr/bin/env python
# coding: utf-8

# # Neural Networks (Continued)

# In[ ]:


from Week11M import *


# Usually we don’t build neural networks by hand. This is in part because we use them
# to solve much bigger problems—an image recognition problem might involve hundreds
# or thousands of neurons. And it’s in part because we usually won’t be able to
# “reason out” what the neurons should be.
# 
# Instead (as usual) we use data to *train* neural networks. The typical approach is an
# algorithm called *backpropagation*, which uses gradient descent or one of its variants.

# ## Backpropagation

# Imagine we have a training set that consists of input vectors and corresponding target
# output vectors.
# 
# For example, in our previous `xor_network` example, the input vector
# [1, 0] corresponded to the target output [1]. Imagine that our network has some set
# of weights. We then adjust the weights using the following algorithm:
# 
# 1. Run `feed_forward` on an input vector to produce the outputs of all the neurons in the network.
# 2. We know the target output, so we can compute a `loss` that’s the sum of the squared errors.
# 3. Compute the gradient of this loss as a function of the output neuron’s weights.
# 4. “Propagate” the gradients and errors backward to compute the gradients with respect to the hidden neurons’ weights.
# 5. Take a gradient descent step.
# 
# Typically we run this algorithm many times for our entire training set until the network
# converges.
# 
# The actual math involved can be quite tedious and requires some solid background in multivariate calculus, especially chain rule, as well as familiarity with matrix and vector calculus.
# 
# For our purposes, the results from computing the gradients layer by layer are as follows:

# In[ ]:


def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Given a neural network, an input vector, and a target vector,
    make a prediction and compute the gradient of the squared error
    loss with respect to the neuron weights.
    """
    # forward pass
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         np.dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]


# Now that we have the gradients, we can train neural networks. Let’s
# try to learn the XOR network we previously designed by hand.
# 
# We’ll start by generating the training data and initializing our neural network with
# random weights:

# In[ ]:


import random
random.seed(0)

# training data
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [[0], [1], [1], [0]]

# start with random weights
network = [ # hidden layer: 2 inputs -> 2 outputs
            [[random.random() for _ in range(2 + 1)],   # 1st hidden neuron
             [random.random() for _ in range(2 + 1)]],  # 2nd hidden neuron
            # output layer: 2 inputs -> 1 output
            [[random.random() for _ in range(2 + 1)]]   # 1st output neuron
          ]


# As usual, we can train it using gradient descent. One difference from our previous
# examples is that here we have several parameter vectors, each with its own gradient,
# which means we’ll have to call `gradient_step` for each of them.

# In[ ]:


from Week10TR import gradient_step
import tqdm


# In[ ]:


learning_rate = 1.0

for epoch in tqdm.trange(20000, desc="neural net for xor"):
    for x, y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)

        # Take a gradient step for each neuron in each layer
        network = [[gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                   for layer, layer_grad in zip(network, gradients)]


# In[ ]:


# check that it learned XOR

print(f"1 XOR 1 = {feed_forward(network, [1, 1])[-1][0]}")
print(f"0 XOR 1 = {feed_forward(network, [0, 1])[-1][0]}")
print(f"1 XOR 0 = {feed_forward(network, [1, 0])[-1][0]}")
print(f"0 XOR 0 = {feed_forward(network, [0, 0])[-1][0]}")


# In[ ]:


print(network)


# ## Example: Fizz Buzz

# The VP of Engineering wants to interview technical candidates by making them solve
# “Fizz Buzz,” the following well-trod programming challenge:
# 
# ```
# Print the numbers 1 to 100, except that if the number is divisible
# by 3, print "fizz"; if the number is divisible by 5, print "buzz";
# and if the number is divisible by 15, print "fizzbuzz".
# ```
# 
# He thinks the ability to solve this demonstrates extreme programming skill. You think
# that this problem is so simple that a neural network could solve it.
# 
# Neural networks take vectors as inputs and produce vectors as outputs. As stated, the
# programming problem is to turn an integer into a string. So the first challenge is to
# come up with a way to recast it as a vector problem.
# 
# For the outputs it’s not tough: there are basically four classes of outputs, so we can
# encode the output as a vector of four 0s and 1s:

# In[ ]:


def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


# In[ ]:


print(fizz_buzz_encode(2))  # [1, 0, 0, 0]
print(fizz_buzz_encode(6))  # [0, 1, 0, 0]
print(fizz_buzz_encode(10)) # [0, 0, 1, 0]
print(fizz_buzz_encode(30)) # [0, 0, 0, 1]


# We’ll use this to generate our target vectors. The input vectors are less obvious.
# 
# You
# don’t want to just use a one-dimensional vector containing the input number, for a
# couple of reasons.
# - A single input captures an “intensity,” but the fact that 2 is twice as much as 1, and that 4 is twice as much again, doesn’t feel relevant to this problem.
# - Additionally, with just one input the hidden layer wouldn’t be able to compute very interesting features, which means it probably wouldn’t be able to solve the problem.
# 
# It turns out that one thing that works reasonably well is to convert each number to its
# binary representation of 1s and 0s (the exact reasoning for this is not obvious, and we don't need to dig into it either).

# In[ ]:


def binary_encode(x: int) -> Vector:
    binary: List[float] = []

    for i in range(10):
        binary.append(x % 2)
        x = x // 2

    return binary


# In[ ]:


#                             1  2  4  8 16 32 64 128 256 512
assert binary_encode(0)   == [0, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(1)   == [1, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(10)  == [0, 1, 0, 1, 0, 0, 0, 0,  0,  0]
assert binary_encode(101) == [1, 0, 1, 0, 0, 1, 1, 0,  0,  0]
assert binary_encode(999) == [1, 1, 1, 0, 0, 1, 1, 1,  1,  1]


# As the goal is to construct the outputs for the numbers 1 to 100, it would be cheating
# to train on those numbers. Therefore, we’ll train on the numbers 101 to 1,023 (which
# is the largest number we can represent with 10 binary digits):

# In[ ]:


xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]


# Next, let’s create a neural network with random initial weights. It will have 10 input
# neurons (since we’re representing our inputs as 10-dimensional vectors) and 4 output
# neurons (since we’re representing our targets as 4-dimensional vectors). We’ll give it
# 25 hidden units, but we’ll use a variable for that so it’s easy to change:

# In[ ]:


NUM_HIDDEN = 25

network = [
    # hidden layer: 10 inputs -> NUM_HIDDEN outputs
    [[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)],

    # output_layer: NUM_HIDDEN inputs -> 4 outputs
    [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]
]


# That’s it. Now we’re ready to train. Because this is a more involved problem (and
# there are a lot more things to mess up), we’d like to closely monitor the training process.
# In particular, for each epoch we’ll track the sum of squared errors and print
# them out. We want to make sure they decrease:

# In[ ]:


learning_rate = 1.0

with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = feed_forward(network, x)[-1]
            error = np.subtract(y, predicted)
            epoch_loss += np.dot(error, error)
            gradients = sqerror_gradients(network, x, y)

            # Take a gradient step for each neuron in each layer
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                    for layer, layer_grad in zip(network, gradients)]

        t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")


# This will take a while to train, but eventually the loss should start to bottom out.
# 
# At last we’re ready to solve our original problem. We have one remaining issue. Our
# network will produce a four-dimensional vector of numbers, but we want a single
# prediction. We’ll do that by taking the `argmax`, which is the index of the largest value:

# In[ ]:


def argmax(xs: list) -> int:
    """Returns the index of the largest value"""
    return max(range(len(xs)), key=lambda i: xs[i])


# In[ ]:


print(argmax([0, -1]))               # items[0] is largest
print(argmax([-1, 0]))               # items[1] is largest
print(argmax([-1, 10, 5, 20, -3]))   # items[3] is largest


# Now we can finally solve “FizzBuzz”:

# In[ ]:


num_correct = 0

for n in range(1, 101):
    x = binary_encode(n)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizz_buzz_encode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])

    if predicted == actual:
        num_correct += 1

print(num_correct, "/", 100)


# So the neural network is able to perform quite well on this particular problem. The performance could be potentially improve if we train more for more epochs.
