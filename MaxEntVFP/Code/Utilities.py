import time
import pickle
from scipy.stats import norm, truncnorm
import numpy as np

def unpack_values (values):
    """
    Takes a tuple with three values corresponding to the 2.5%, 50% and 97.5% quantiles and returns:
    - The median value (50% quantile)
    - A tuple containing the 2.5 and 5% quantiles, useful for setting bounds
    - A normal distribution with the correct standard deviation, truncated to the 95% confidence
      interval
    """
    std_norm_bounds = (-1.96,1.96) # 1.96 is the approximate value of the 97.5 percentile point of the
                                   # standard normal distribution. 95% of the area under a normal curve
                                   # lies within roughly 1.96 standard deviations of the mean.
    return values[1], \
          (values[0], values[2]), \
           truncnorm(*std_norm_bounds, loc=values[1], scale=(values[2]-values[0])/(2*1.96))

def pretty_ptemcee(fitter, nsamples, nthin, name=None, nCPUs=-1, save=True):
    """
    Makes the PT-MCMC sampling slightly easier to keep track of. Also pickles the fitter for you
    every nthin samples to reduce data loss.
    
    parameters:
    fitter : refnx.analysis.CurveFitter
        the fitter that will be sampled from
    nsamples : int
        The number of samples you want to end up with (total number of steps will be nsamples*nthin)
    nthin : int
        The number of steps between each sample
    name : String (default None)
        Name for saving the fitter. If none uses the name attached to fitter.objective
    nCPUs : int (default -1)
        Number of threads to sample with. If -1 will create threads equal to the number of CPU cores    
    """
    objective = fitter.objective

    if name is None:
        name = objective.name

    t = time.strftime('%d/%m %H:%M: ')
    print('%s fitting: %s' % (t, name))

    for i in range(nsamples):
        time.sleep(1) # For some reason this stops the fitter creating two progress bars
        fitter.sample(1, nthin=nthin, pool=nCPUs)
        time.sleep(1)

        average_logpost = np.mean(fitter.logpost[:, 0], axis=1)
        if fitter.chain.shape[0] > 1:
            diff = np.diff(average_logpost)/nthin
        else:
            diff = [0]

        t = time.strftime('%d/%m %H:%M: ')
        print("%s %d/%d - logpost: %d  dlogpost/dstep: %0.2f" %
              (t, i+1, nsamples, average_logpost[-1], diff[-1]))

        if save:
            pickle.dump(fitter, open(name + '_fitter.pkl', 'wb'))