import numpy as np
import matplotlib.pyplot as plt

"""Generate a set of n datapoints from the function in interval
[start, end] with given additive noise (normal distributed with given std-dev)
"""
def generate_datapoints(n, start=0, end=1, func=(lambda x: x), noise=1):
    rng = np.random.default_rng(12345)
    data = np.ndarray((2,n), dtype = float)
    data[0,:] = rng.random(n)*(end-start)+start
    data[1,:] = func(data[0,:])

    if noise:
        data[1,:] += rng.normal(0, noise, n)

    return data

def show_datapoints(x, y):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.scatter(x, y)
    plt.show()

def main():
    data = generate_datapoints(n=100, start=-10, end=10, func=(lambda x: x*x), noise=0.8)
    #show_datapoints(data[0,:], data[1,:])




if __name__ == "__main__":
    main()