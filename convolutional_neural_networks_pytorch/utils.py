import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class DataGenerator(object):
    """
    Utility class to randomly generate data for regression (linear and polynomial)
    and classification (set of normal distributions).
    
    It allows to load some UCI datasets (e.g. iris, boston).

    Example
    -------
        from utils import DataGenerator

        dg = DataGenerator()

        # create a random linear dataset
        dataset = dg.generate_linear_data(100, 2, 1)

        # read UCI-IRIS
        trainset, testset = dg.load("iris")
    """


    def __init__(self):
        pass


    def generate_linear_data(self, nsamples, slope, offset, noise_scale=0.02):
        """
        Generate a set of float32 points organized as a 2D matrix.
        Data format is NxF where N is the number of samples and F the number of features.

        Each sample is picked from a noisy linear model.

        Parameters
        ----------
        nsamples : integer
            Number of samples. Minimum is 2.
        slope : float
            Slope of the linear model.
        offset : float
            Offset of the linear model.
        noise_scale : float, optional (default=0.02)
            Scale of the noise between 0 and 1.

        Returns
        -------
        (trainset, testset)
            Split dataset. Both are in the Mx2 format.
        """

        assert isinstance(nsamples, int) and nsamples > 1
        assert isinstance(slope, float)
        assert isinstance(offset, float)
        assert isinstance(noise_scale, float) and 0. <= noise_scale <= 1.

        # randomly pick a bunch of points between 0 and 1
        x_vals = np.random.rand(nsamples)
        # generate random noise
        noise = noise_scale * np.random.normal(loc=0., scale=1., size=(nsamples,1)).astype(np.float32)

        # create the complete dataset as float32 values
        x_data = np.asarray(x_vals, dtype=np.float32).reshape(-1,1)
        y_data = np.asarray([slope * x + offset for x in x_vals], dtype=np.float32).reshape(-1,1) + noise

        # concatenate the features/targets in the format Nx2
        data = np.concatenate((x_data, y_data), axis=-1)

        # randomly split the dataset through sklearn utility
        trainset, testset = train_test_split(data, test_size=0.25)

        return trainset, testset


    def generate_polynomial_data(self, nsamples, polydeg, noise_scale=0.02):
        """
        Generate a set of float32 points organized as a 2D matrix.
        Data format is NxF where N is the number of samples and F the number of features.

        Each sample is picked from a noisy polynimial.

        Parameters
        ----------
        nsamples : integer
            Number of samples. Minimum is 2.
        polydeg : integer
            Maximum degree of the polynomial. Minimum is 1.
        noise_scale : float, optional (default=0.02)
            Scale of the noise between 0 and 1.

        Returns
        -------
        (trainset, testset)
            Split dataset. Both are in the Mx2 format.
        """

        assert isinstance(nsamples, int) and nsamples > 1
        assert isinstance(polydeg, int) and polydeg > 0
        assert isinstance(noise_scale, float) and 0. <= noise_scale <= 1.

        # randomly pick a bunch of points between 0 and 1
        x_vals = np.random.rand(nsamples)
        # generate random noise
        noise = noise_scale * np.random.normal(loc=0., scale=1., size=(nsamples,1)).astype(np.float32)

        # +1 on polydeg is for the offset
        coeffs = 2 * np.random.rand(polydeg + 1) - 1

        # create the complete dataset as float32 values
        x_data = np.asarray(x_vals, dtype=np.float32).reshape(-1,1)
        for deg in range(2, polydeg + 1):
            x_data = np.concatenate( (np.power(x_data[:,-1].reshape(-1,1), deg), x_data) , axis=-1)
        x_data = np.concatenate( (x_data, np.ones((nsamples,1))), axis=-1 )
        y_data = np.sum(coeffs * x_data, axis=-1, dtype=np.float32).reshape(-1,1) + noise
        x_data = x_data[:,-2].astype(np.float32).reshape(-1,1)

        # concatenate the features/targets in the format Nx2
        data = np.concatenate((x_data, y_data), axis=-1)

        # randomly split the dataset through sklearn utility
        trainset, testset = train_test_split(data, test_size=0.25)

        return trainset, testset


    def generate_gaussian_data(self, nclasses, nsamples, spread_factor=1.):
        """
        Generate a set of float32 points organized as a 2D matrix.
        Data format is NxF where N is the number of samples and F the number of features.

        Each sample is picked from a random normal distribution.

        Parameters
        ----------
        nclasses : integer
            Number of classes. Minimum is 2.
        nsamples : integer
            Number of samples per class. Minimum is 2.
        spread_factor : float
            Define the distance among the class point distributions.

        Returns
        -------
        (trainset, testset)
            Split dataset. Both are in the Mx2 format.
        """

        assert isinstance(nclasses, int) and nclasses > 1
        assert isinstance(nsamples, int) and nsamples > 1

        x_data = None
        y_data = None

        for cid in range(nclasses):
            mu = spread_factor * np.random.rand(2) - 1
            sd = np.random.rand(2,2) - 1
            points = np.random.multivariate_normal(mean=mu, cov=sd, size=(nsamples))
            target = np.ones((nsamples,1)) * cid
            if x_data is None:
                x_data = points
                y_data = target
                continue
            x_data = np.concatenate((x_data, points))
            y_data = np.concatenate((y_data, target))

        # concatenate the features/targets in the format Nx2
        data = np.concatenate((x_data, y_data), axis=-1)
        data = data.astype(np.float32)

        # randomly split the dataset through sklearn utility
        trainset, testset = train_test_split(data, test_size=0.25)

        return trainset, testset


    def _load_notmnist(self):
        """
        Load notMNIST dataset
        """

        data = np.load("notMNIST_small.npy", allow_pickle=True)
        data = data.any()

        nsamples = data["images"].shape[0]

        dataset = {}
        dataset["data"] = data["images"].reshape(nsamples, -1)
        dataset["target"] = data["labels"]

        return dataset


    def load(self, name):
        """
        Load a UCI dataset from sklearn module.

        Parameters
        ----------
        name : string
            Name of the dataset. Currently supported:
            [classification]    [regression]        
            - iris              - boston            
            - digits
            - notmnist

        Returns
        -------
        (trainset, testset)
            Split dataset. Both are in the MxF format.
        """

        if name == "iris":
            data = datasets.load_iris()
        elif name == "digits":
            data = datasets.load_digits()
        elif name == "notmnist":
            data = self._load_notmnist()
        elif name == "boston":
            data = datasets.load_boston()
        else:
            raise NotImplementedError()
        
        x_data = data["data"]
        y_data = data["target"].reshape(-1,1)
        data = np.concatenate((x_data,y_data), axis=-1)
        data = data.astype(np.float32)
        trainset, testset = train_test_split(data, test_size=0.25)

        return trainset, testset