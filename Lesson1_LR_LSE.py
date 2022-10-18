"""
# Exercise 0) 
#  - Run the script and load the boston dataset
#  - Look the EDA on the boston dataset
#
# Exercise 1)
#  - Implement the `LSRegression` function
#  - Use its output (W) to compute the projection for the input data X
#  - See visualization of values and hyperplane
#
# Exercise 2)
#  - Implement the `MyGDregression` function and `gradfn` as we have seen during the class.
#  - Use its output (W) to compute the projection for the input data X
#  - See visualization of values and hyperplane
#
# Exercise 3)
#  - Proof that Gradient Descent is faster w.r.t. Least Squares (hint: watch the sklearn function `make_regression`)
#

"""
import time
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

from scipy import stats
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Global parameters
visualize = False
features = ["LSTAT", "RM"]
target_name = "MEDV"


def inspect_boston_dataset(
    visualize: bool = True,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    boston = load_boston()
    boston_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_pd[target_name] = boston.target

    if visualize:
        sns.set(rc={"figure.figsize": (11.7, 8.27)})
        sns.distplot(boston_pd[target_name], bins=30)
        # sns.histplot(boston_pd[target_name], kde=True, stat="density", linewidth=0)
        plt.show(block=True)
        plt.close()
    # time.sleep(1)

    X = np.array(boston.data, dtype="f")
    Y = np.array(boston.target, dtype="f")

    if visualize:
        fig, axs = plt.subplots(7, 2, figsize=(14, 30))
        for index, feature in enumerate(boston.feature_names):
            subplot_idx = int(index / 2)
            if index % 2 == 0:
                axs[subplot_idx, 0].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 0].set_xlabel(feature)
                axs[subplot_idx, 0].set_ylabel("Target")
            else:
                axs[subplot_idx, 1].scatter(x=X[:, index], y=Y)
                axs[subplot_idx, 1].set_xlabel(feature)
                axs[subplot_idx, 1].set_ylabel("Target")
        # plt.savefig("linearity_scatter_plots.png")
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    if visualize:
        correlation_matrix = boston_pd.corr().round(2)
        # annot = True to print the values inside the square
        sns.heatmap(data=correlation_matrix, annot=True)
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    if visualize:
        target = boston_pd[target_name]
        plt.figure(figsize=(20, 5))
        for i, col in enumerate(features):
            plt.subplot(1, len(features), i + 1)
            x = boston_pd[col]
            y = target
            plt.scatter(x, y, marker="o")
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel(target_name)
        # plt.savefig('sel_features_analysis.png')
        plt.show(block=True)
        plt.close()
        # time.sleep(1)

    X = boston_pd[features]
    Y = boston_pd[target_name]

    return boston, boston_pd, X, Y


def plot3d_lr(W, X, title):
    # create x,y,z data for 3d plot
    y_pred = np.sum(W[0:-1] * X, axis=1) + W[-1]
    if isinstance(X, pd.DataFrame):
        x = X[features[0]]
        y = X[features[1]]
    else:
        x = X[:, 0]  # LSTAT
        y = X[:, 1]  # RM
    z = y_pred
    data = np.c_[x, y, z]

    # Create X, Y data to predict
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    XX, YY = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))

    # calculate prediction
    Z = W[0] * XX + W[1] * YY + W[-1]
    # plot the surface
    fig = plt.figure()
    # ax = fig.gca(projection="3d")
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="r", s=50)
    plt.title(title)
    plt.xlabel(features[0])  # LSTAT
    plt.ylabel(features[1])  # RM
    ax.set_zlabel(target_name)
    plt.show()
    # plt.savefig('3d_plane_of_best_fit.png')
    plt.close()


def LSRegression(X, y):
    # ------------------ Least Squares Estimation ------------------
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    W = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))

    return W


def perform_lr_ls(X, Y):
    """
    Given `X` - matrix of shape (N,D) of input features
          `Y` - target y values
    Solves for linear regression using the Least Squares algorithm. implemented in LSRegression
    Returns weights and prediction.
    """
    W = LSRegression(X, Y)
    Y_pred = np.dot(X, W[0:-1
    ]) + W[-1]

    plt.figure(figsize=(4, 3))
    plt.scatter(Y, Y_pred)
    # plt.plot([0, 50], [0, 50], '--k')
    plt.axis("tight")
    plt.grid()
    plt.title("LS solution")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    plot3d_lr(W, X, "LS LR")
    return W, Y_pred


# SKLEARN Linear Regression model
def perform_lr_sklearn(X, Y):
    lr_sk = LinearRegression()
    lr_sk.fit(X, Y)
    W = np.hstack((lr_sk.coef_, lr_sk.intercept_))
    Y_pred = lr_sk.predict(X)
    loss_sk = mean_squared_error(Y, Y_pred)
    print("Model performance SKLEARN LR:")
    print("--------------------------------------")
    print("MSE is {}".format(loss_sk))
    print("\n")

    plt.figure(figsize=(4, 3))
    plt.scatter(Y, Y_pred)
    # plt.plot([0, 50], [0, 50], '--k')
    plt.axis("tight")
    plt.grid()
    plt.title("SKLearn Linear")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    plot3d_lr(W, X, "SKLearn")

    return W, Y_pred


# Gradient Descent


def gradfn(theta, X, Y):
    """
    Given `theta` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,D) of input features
          `t` - target y values
    Return gradient of each weight evaluated at the current value
    """
    delta_theta = np.zeros_like(theta)

    delta_theta = (Y - np.dot(X,theta))

    return delta_theta

def MyGDregression(X, Y, niter, alpha):
    """
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    """
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    rng = np.random.default_rng()
    W = rng.random(X.shape[1])

    for i in np.arange(niter):
        W = W - alpha*np.dot(X.transpose(),np.dot(X,W)-Y)/len(Y)

    return W

def perform_lr_gd(X, Y, iters: int = 10000, alpha: float = 0.005):

    theta = MyGDregression(X, Y, iters, alpha)
    # what should Y_pred_GD be?
    Y_pred_GD = np.dot(X,theta[0:-1]) + theta[-1]

    loss_sgd = mean_squared_error(Y, Y_pred_GD)
    print("Model performance GD:")
    print("--------------------------------------")
    print("MSE is {}".format(loss_sgd))
    print("\n")

    plt.figure(figsize=(4, 3))
    plt.scatter(Y, Y_pred_GD)
    # plt.plot([0, 50], [0, 50], '--k')
    plt.axis("tight")
    plt.grid()
    plt.title("GD")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    plot3d_lr(theta, X, "GD")

    return theta, Y_pred_GD


def residual_analysis(Y, Y_pred):

    # Now that we have the results, we can also analyze the residuals
    residuals = Y - Y_pred

    # First thing we can easily do is check the residuals distribution.
    # We expect to see a normal or "mostly normal" distribution.
    # For this we can use a histogram
    # ax = sns.distplot(residuals, kde=True)
    ax = sns.histplot(residuals, kde=True, stat="density", linewidth=0)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")
    plt.title("Residuals distribution")
    # plt.savefig("regression_residual_kde.png")
    plt.show()
    plt.close()

    # Lastly, we can check if the residuals behave correctly with a normal probability plot
    # This can be very useful when we have few samples, as this graph is more sensitive than the histogram.
    # We can easily calculate the theoretical values needed for the normal probability plot.
    fig = plt.figure()
    res = stats.probplot(residuals, plot=plt, fit=False)
    # plt.savefig("residual_normality_plot.png")
    plt.show()
    plt.close()

def main():
    boston_raw, boston_pd, X, Y = inspect_boston_dataset(visualize)


    # Data non-normalized
    W, Y_pred = perform_lr_ls(X, Y)
    #residual_analysis(Y, Y_pred)
    print("R2 score", r2_score(Y, Y_pred))

    W, Y_pred = perform_lr_sklearn(X, Y)
    #residual_analysis(Y, Y_pred)
    print("R2 score", r2_score(Y, Y_pred))

    T, Y_pred = perform_lr_gd(X, Y, iters=10000, alpha=0.005)
    # residual_analysis(Y, Y_pred)
    print("R2 score", r2_score(Y, Y_pred))

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Data normalized
    W_norm, Y_pred_norm = perform_lr_ls(X, Y)
    # residual_analysis(Y, Y_pred_norm)
    print("R2 score", r2_score(Y, Y_pred_norm))

    W_norm, Y_pred_norm = perform_lr_sklearn(X, Y)
    # residual_analysis(Y, Y_pred_norm)
    print("R2 score", r2_score(Y, Y_pred_norm))

    T_norm, Y_pred_norm = perform_lr_gd(X, Y)
    # residual_analysis(Y, Y_pred_norm)
    print("R2 score", r2_score(Y, Y_pred_norm))

    print(T, T_norm)

    # ----------------------- LS vs GD -----------------------

    # TODO

if __name__ == "__main__":
    main()
