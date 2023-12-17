# Task 4 -- Reinforcement Learning

## Problem

[Task description](task4_description.pdf)

[Supplementary material](task4_supplementary.pdf)

## Code

[Python file](solution.py)

## Report

For the solution of this project, we first implemented the "vanilla" version of SAC (with fixed entropy regularization coefficient alpha). For the MLP (class NeuralNetwork) we used the following structure: First we have a linear layer that goes from input_dim to hidden_size and an activation layer. Then we have #hidden_layers of linear layer + activation layer. In the end, we have one more linear layer which makes sure the output is output_dim. The activations can be either Tanh or ReLU.

The Critic is made up by Q1 and Q2 networks, which are instantiations of the network structure above. They are optimized over using a single optimizer (Adam) with some specified learning rate. The critic target is updated using a tau if soft update is specified in the arguments.

For the Actor we have implemented two policies: GaussianPolicy and DeterministicPolicy. The deterministic policy is relatively straight forward, whereas the Gaussian policy was harder to implement. We use the reparameterization trick and then enforce the action bound.

In each step of training the agent, we perform 10 iterations. In each of them, we update the critic target again (soft update using tau). In the end, we changed our code to implement a trainable parameter for the alpha, varying it over the course of training.