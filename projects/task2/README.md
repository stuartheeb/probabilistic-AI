# Task 2 -- SWA-Gaussian

## Problem

[Task description](task2_description.pdf)

## Code

Located in directory `handin`

[Python file](handin/solution.py)

## Report

Given the code template, we implemented SWA-Gaussian (SWAG), as it is described in the paper. In the first step, we implemented SWAG-Diagonal, which does not yet make use of the low-rank part. This first part was actually a lot more work than expected. We had many bugs at first and had to figure out how to save the first and second moments in a way so that we could do the sampling part easily. Sampling z1 as a Gaussian was not too difficult. A bit more difficult was the sampling of z2 and extending our solution to "Full SWAG", especially multiplying D_hat and z2 in an efficient way (the tip to use the deque was useful). We use the same scale as in the paper, but only use it when computing full SWAG. We update the batchnorm after sampling (once per function call, not inside the loop). For prediction, we had considered using a separate prediction threshold for each of the classes, but in the end we ran out of time to try that. After implementing SWAG, we grid-searched in order to tune some of the hyperparameters. We chose the following: swag epochs, swag learning rate, bma samples, deviation matrix max. rank and prediction threshold. The only values that changes by a fair amount by grid-searching were the swag learning rate and the prediction threshold, which made sense to use due to the fact that there is a penalty when predicting a class in an ambiguous case.