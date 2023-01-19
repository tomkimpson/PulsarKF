#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data consisting of a Gaussian with a mean and variance
"""
import bilby
import numpy as np






if __name__=="__main__":




    # A few simple setup steps
    label = "gaussian_example"
    outdir = "outdir"

    # Here is minimum requirement for a Likelihood class to run with bilby. In this
    # case, we setup a GaussianLikelihood, which needs to have a log_likelihood
    # method. Note, in this case we will NOT make use of the `bilby`
    # waveform_generator to make the signal.

    # Making simulated data: in this case, we consider just a Gaussian

    data = np.random.normal(3, 4, 100)

    from bug_class import SimpleGaussianLikelihood


    likelihood = SimpleGaussianLikelihood(data)
    priors = dict(
        mu=bilby.core.prior.Uniform(0, 5, "mu"),
        sigma=bilby.core.prior.Uniform(0, 10, "sigma"),
    )

    # And run sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=1000,
        outdir=outdir,
        label=label,
        maxmcmc=2000,
        check_point=False,
        resume=False,
        npool=1
    )
    
