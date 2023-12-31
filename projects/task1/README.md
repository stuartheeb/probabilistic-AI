# Task 1 -- Gaussian Processes

## Problem

[Task description](task1_description.pdf)

## Code

[Python file](solution.py)

## Report

For the first task, we are asked to predict the concentration of fine particulate matter (pm25). The data is given by lat/lon positions and corresponding pm25 concentrations, for each of which it is indicated whether is it a candidate location for new residential areas (1) or not (0). This indicator is very important, because the loss is computed as a weighted sum of squared differences, where there is a weight penalty factor of 50 when we *under*estimate a potential residential area, and a unit weight otherwise.

As preprocessing, we fill out the skeleton function simply by dividing up the original train_x and train_y data into the desired columns, separating lat/lon from the area data.

As our first approach, we implemented a solution with sklearn's Gaussian Process Regression (GPR). In a first step, we standardized the training data, and experimented with also standardizing the training labels. Of course, at time of prediction, we transformed the predictions back to the original scale. We optimized this model, trying different kernels (mainly RBF and its generalization, the Matern kernel) with different parameters. On top of this, we implemented a clustering scheme using KMeans and then trained local GPs to each cluster. We were able to get decent scores, but we soon realized that using the library GPyTorch was able to get better scores with much a much simpler implementation:

For our final approach, we use the ExactGP model from the GPyTorch library. The approach follows closely to what is presented in the GPyTorch documentation. Upon further inspection we realized that using the ConstantMean as the mean module and a composition of the ScaleKernel and RBFKernel for the covariance module for the ExactGP worked quite well, which is also true for the likelihood being the GaussianLikelihood. 

To counteract the penalty effect of the cost function we decided to translate our predictions by an amount (corrective_factor)*(gp_std) (the stddev we get from the GP) for all test samples that have area_id = 1. We experimented with some values for the corrective factor and ended with 0.75 as a good option.

Since the training process was much quicker run directly on the machine, the model was trained running the python script directly and then saved as a .pth file, which was then loaded using torch.load() when run in the Docker to produce the final result.