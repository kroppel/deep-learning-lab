import numpy as np
import matplotlib.pyplot as plt
import time

"""Generate a set of n datapoints from the function in interval
[start, end] with given additive noise (normal distributed with given std-dev)
"""
def generate_datapoints(number_dp, dim_x, start=0, end=1, func=(lambda x: x), noise=1):
    rng = np.random.default_rng(int(time.time()))

    X = rng.random((number_dp, dim_x))*(end-start)+start
    print(X.shape)
    Y = np.apply_along_axis(func, 1, X)
    Y = Y.reshape((number_dp,1))
    Y = Y + rng.normal(0, noise, (number_dp, 1))

    return X, Y

"""Perform linear regression using least squares method
"""
def LSRegressionLinear(X, y):
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    W = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))

    return W

"""Perform quadratic regression using least squares method
"""
def LSRegressionQuadratic(X, y):
    X_mod = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    # compute outer product between data vectors in X
    X_outer = np.einsum('...i,...j->...ij',X,X)
    # choose upper right part of pairwise products of the components of x
    # (including diagonal) to concatenate to vector x
    for dim in np.arange(1, X.shape[1]+1):
        X_mod = np.concatenate((X_outer[:,-dim,-dim:],X_mod), axis=1)
    #X_mod = np.concatenate((np.power(X, 2), X_mod), axis=1)
    W = np.dot(np.linalg.inv(np.dot(X_mod.transpose(), X_mod)), np.dot(X_mod.transpose(), y))

    return W

"""Plot sets of data points given by (x,y)
"""
def show_datapoints(x, y):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    axs.spines['bottom'].set_position('zero')
    axs.spines['left'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.scatter(x, y)
    plt.show()

"""Plot sets of data points given by (x,y) and
the regression model func with its parameters w
"""
def show_datapoints_and_regression(x, y, func, w):
    # sort datapoints (important for line plot)
    sorted_indices = np.argsort(x, axis=0)
    x = np.take_along_axis(x, sorted_indices, axis=0)
    y = np.take_along_axis(y, sorted_indices, axis=0)
    
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    axs.spines['bottom'].set_position('zero')
    axs.spines['left'].set_position('zero')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.scatter(x, y)
    axs.plot(x, (func(w))(x))
    plt.show()

"""Performs the outer product for each row vector in matrix X
with itself and returns the upper right result matrix triangle
including the diagonal
"""
def add_X_outer1D(X):
    if len(X.shape)==1:
        # compute outer product between data vectors in X
        X_outer = np.einsum('...i,...j->...ij',X,X)
        # choose upper right part of pairwise products of the components of x
        # (including diagonal) to concatenate to vector x
        for dim in np.arange(1, X.shape[0]+1):
            X = np.concatenate((X_outer[-dim,-dim:],X), axis=0)
    else:   
        # compute outer product between data vectors in X
        X_outer = np.einsum('...i,...j->...ij',X,X)
        # choose upper right part of pairwise products of the components of x
        # (including diagonal) to concatenate to vector x
        for dim in np.arange(1, X.shape[1]+1):
            X = np.concatenate((X_outer[:,-dim,-dim:],X), axis=1)

    return X
    
def add_X_outerMD(X):
    if len(X.shape)==1:
        # compute outer product between data vectors in X
        X_outer = np.einsum('...i,...j->...ij',X,X)
        # choose upper right part of pairwise products of the components of x
        # (including diagonal) to concatenate to vector x
        for dim in np.arange(1, X.shape[1]+1):
            X = np.concatenate((X_outer[:,-dim,-dim:],X), axis=1)
    else:   
        # compute outer product between data vectors in X
        X_outer = np.einsum('...i,...j->...ij',X,X)
        # choose upper right part of pairwise products of the components of x
        # (including diagonal) to concatenate to vector x
        for dim in np.arange(1, X.shape[1]+1):
            X = np.concatenate((X_outer[:,-dim,-dim:],X), axis=1)

    return X

def model_linear(weights):
    return lambda x: np.dot(x,weights[0:-1])+weights[-1]

def model_quadratic1D(weights):  
    return lambda x: np.dot(np.power(x,2),weights[0])+np.dot(x,weights[1])+weights[-1]

def model_quadraticMD(weights):  
    return lambda x: np.dot(add_X_outerMD(x),weights[0:-1])+weights[-1] if len(x.shape)!=1 \
        else np.dot(add_X_outer1D(x),weights[0:-1])+weights[-1]

def linear_regression_univariate_example():
    # Generate linear data
    X, Y = generate_datapoints(number_dp=100, dim_x=1, start=-10, end=10, func=model_linear(np.asarray([2, 10])), noise=0.8)
    show_datapoints(X, Y)
    # Perform linear regression
    weights = LSRegressionLinear(X, Y)
    predictions = np.dot(X, weights[0:-1]) + weights[-1]
    show_datapoints_and_regression(X, Y, model_linear, weights)
    mse = np.sum(np.power(predictions-Y, 2))/len(Y)

    print("MSE: "+str(mse))
    print(weights)

def linear_regression_multivariate_example():
    # Generate linear data
    X, Y = generate_datapoints(number_dp=100, dim_x=3, start=-10, end=10, func=model_linear(np.asarray([2, -3, 4, -5])), noise=1)
    # Perform linear regression
    weights = LSRegressionLinear(X, Y)
    predictions = np.dot(X, weights[0:-1]) + weights[-1]
    mse = np.sum(np.power(predictions-Y, 2))/len(Y)

    print("MSE: "+str(mse))
    print(weights)

def quadratic_regression_univariate_example():
    # Generate quadratic data
    X, Y = generate_datapoints(number_dp=100, dim_x=1, start=-10, end=10, func=model_quadratic1D(np.asarray([0.8, 5, -5])), noise=3)
    show_datapoints(X, Y)
    # Perform linear regression
    weights = LSRegressionQuadratic(X, Y)
    print(weights)

    X_mod = add_X_outer1D(X)

    predictions = np.dot(X_mod, weights[0:-1]) + weights[-1]
    show_datapoints_and_regression(X, Y, model_quadratic1D, weights)
    mse = np.sum(np.power(predictions-Y, 2))/len(Y)

    print("MSE: "+str(mse))
    print(weights)

def quadratic_regression_multivariate_example():
    # Generate quadratic data (#parameters = sum([1,2,...,dim_x, dim_x+1]))
    X, Y = generate_datapoints(number_dp=100, dim_x=2, start=-10, end=10, func=model_quadraticMD(np.asarray([0.8, 4, 3, 2, 2, -5])), noise=3)
    #show_datapoints(X, Y)
    # Perform linear regression
    weights = LSRegressionQuadratic(X, Y)
    print(X.shape)

    X_mod = add_X_outerMD(X)

    predictions = np.dot(X_mod, weights[0:-1]) + weights[-1]
    #show_datapoints_and_regression(X, Y, model_quadratic, weights)q
    mse = np.sum(np.power(predictions-Y, 2))/len(Y)

    print("MSE: "+str(mse))
    print(weights)

def main():
    #linear_regression_univariate_example()
    #linear_regression_multivariate_example()
    quadratic_regression_univariate_example()
    quadratic_regression_multivariate_example()


if __name__ == "__main__":
    main()