# Task 3 -- Bayesian Optimization

## Problem

[Task description](task3_description.pdf)

## Code

[Python file](solution.py)

## Report

To model v, we used a GP with alpha=0.0001^2 (std_dev as given in the task description) and with kernel Const(4)+DotProduct(0)+Matern(nu=2.5, length_scale=0.5).

To model f, we used a GP with alpha=0.15^2 (std_dev as given in the task description) and with kernel Matern(nu=2.5).

The next recommendations function simply returns the result from the optimize_acquisition_funtion().

We define the acquisition_function() with the Upper Confidence Bound (UCB), with beta=4.0 and gamma=100. We use the GPs for v and f to predict their mean and std which we need for UCB.

The function to add data points simply appends x, f, v to their arrays respectively.

For our understanding and for debugging purposes we implemented the function plot() which helped us understand the problem and our results.