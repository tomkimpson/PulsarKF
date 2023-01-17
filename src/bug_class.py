import bilby
import numpy as np
class SimpleGaussianLikelihood(bilby.Likelihood):
        def __init__(self, data):
            """
            A very simple Gaussian likelihood

            Parameters
            ----------
            data: array_like
                The data to analyse
            """
            super().__init__(parameters={"mu": None, "sigma": None})
            self.data = data
            self.N = len(data)

        def log_likelihood(self):
            mu = self.parameters["mu"]
            sigma = self.parameters["sigma"]
            res = self.data - mu
            return -0.5 * (
                np.sum((res / sigma) ** 2) + self.N * np.log(2 * np.pi * sigma**2)
            )