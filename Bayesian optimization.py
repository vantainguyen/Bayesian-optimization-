# Multimodel function optimization

import numpy as np
from numpy import arange
from numpy import asarray
from numpy import vstack
from numpy import argmax 
from numpy.random import normal
from numpy.random import random 
from scipy import sin, pi
from scipy.stats import norm
from matplotlib import pyplot as plt
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

# Adding noise with mean = 0 and standard deviation = 0.1 to the objective function
def objective(x, noise=0.1):
    noise = np.random.normal(loc=0, scale=noise)
    return x**2*sin(5*pi*x)**6 + noise


def surrogate(model, X):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X, return_std = True)
# plot real observations vs surrogate function

def plot(X, y, model):
    # scatter plot of inputs and real objective function 
    plt.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = asarray(arange(0,1,.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples,_ = surrogate(model, Xsamples)
    plt.plot(Xsamples, ysamples)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesian optimization for function y = f(x)')
    # show the plot
    plt.show()
    
    
# sample the domain sparsely with noise
X = random(100)
y = asarray([objective(x) for x in X])

# reshape into rows and cols
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot the surrogate function
plot(X, y, model)

def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:,0]
    # calculate the probability of improvement
    probs = norm.cdf((mu-best)/(std+1e-9))
    return probs
# optimize the acquisition function
def opt_acquisition(X, y, model):
    Xsamples = random(100)
    Xsamples = Xsamples.reshape(-1,1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix,0]

# perform the optimization process
for i in range(200):
    # select the next point to sample
    x = opt_acquisition(X, y, model)
    # sample the point
    actual = objective(x)
    # summarize the finding for our own reporting
    est, _ = surrogate(model, [[x]])
    print('iter %d, >x=%.3f, f()=%3f, actual=%.3f' %(i, x, est, actual))
    # add the data to the dataset
    X = vstack((X, [[x]]))
    y = vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)
    
# plot all samples and the final surrogate function 
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x = %.3f, y = %.3f' %(X[ix], y[ix]))
    










