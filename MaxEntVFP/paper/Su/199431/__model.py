
import numpy
import os
from refl1d.names import *
from math import *
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter('ignore', UserWarning)

# Maximum Q-value ##############################################################
q_min = 0.0
q_max = 1.0

reduced_file = "/SNS/users/raylinc2/__data.txt"

Q, R, dR, dQ = numpy.loadtxt(reduced_file).T
i_min = min([i for i in range(len(Q)) if Q[i]>q_min])
i_max = max([i for i in range(len(Q)) if Q[i]<q_max])+1

# SNS data is FWHM
dQ_std = dQ/2.35
probe = QProbe(Q[i_min:i_max], dQ_std[i_min:i_max], data=(R[i_min:i_max], dR[i_min:i_max]))

# Materials ####################################################################
SiWAFER705V = SLD(name='SiWAFER705V', rho=2.165, irho=0.0)
D2O05V = SLD(name='D2O05V', rho=6.272, irho=0.0)
SiO2 = SLD(name='SiO2', rho=3.004, irho=0.0)
Pd = SLD(name='Pd', rho=4.088, irho=0.0)
void = SLD(name='void', rho=6.094, irho=0.0)
material1 = SLD(name='material1', rho=4.544, irho=0.0)
material2 = SLD(name='material2', rho=5.193, irho=0.0)
material3 = SLD(name='material3', rho=5.497, irho=0.0)
material4 = SLD(name='material4', rho=5.006, irho=0.0)
material5 = SLD(name='material5', rho=5.386, irho=0.0)
material6 = SLD(name='material6', rho=5.318, irho=0.0)
material7 = SLD(name='material7', rho=4.552, irho=0.0)
material8 = SLD(name='material8', rho=5.558, irho=0.0)


# Film definition ##############################################################
sample = (  D2O05V(0, 30.11) | material8(188.0, 30.34) | material7(323.7, 47.32) | material6(567.2, 81.63) | material5(589.2, 94.78) | material4(417.7, 99.87) | material3(422.7, 11.56) | material2(109.2, 14.42) | material1(40.39, 8.757) | void(18.64, 1.548) | Pd(109.9, 2.737) | SiO2(25.02, 5.87) | SiWAFER705V )

sample['SiO2'].thickness.range(25.0, 30.0)
sample['SiO2'].material.rho.range(3.0, 3.4)
sample['SiO2'].interface.range(2.0, 6.0)
sample['Pd'].thickness.range(100.0, 110.0)
sample['Pd'].material.rho.range(3.9, 4.1)
sample['Pd'].interface.range(2.0, 6.0)
sample['void'].thickness.range(5.0, 50.0)
sample['void'].material.rho.range(5.5, 6.3)
sample['void'].interface.range(1.0, 5.0)
sample['material1'].thickness.range(30.0, 600.0)
sample['material1'].material.rho.range(4.5, 6.36)
sample['material1'].interface.range(2.5, 25.0)
sample['material2'].thickness.range(20.0, 600.0)
sample['material2'].material.rho.range(4.5, 6.36)
sample['material2'].interface.range(10.0, 25.0)
sample['material3'].thickness.range(30.0, 600.0)
sample['material3'].material.rho.range(4.5, 6.36)
sample['material3'].interface.range(10.0, 50.0)
sample['material4'].thickness.range(100.0, 600.0)
sample['material4'].material.rho.range(4.5, 6.36)
sample['material4'].interface.range(30.0, 100.0)
sample['material5'].thickness.range(100.0, 600.0)
sample['material5'].material.rho.range(4.5, 6.36)
sample['material5'].interface.range(30.0, 100.0)
sample['material6'].thickness.range(100.0, 600.0)
sample['material6'].material.rho.range(4.5, 6.36)
sample['material6'].interface.range(30.0, 100.0)
sample['material7'].thickness.range(100.0, 600.0)
sample['material7'].material.rho.range(4.5, 6.36)
sample['material7'].interface.range(30.0, 100.0)
sample['material8'].thickness.range(100.0, 600.0)
sample['material8'].material.rho.range(4.5, 6.36)
sample['material8'].interface.range(30.0, 100.0)



probe.intensity=Parameter(value=1.0,name='normalization')
probe.background=Parameter(value=0.0,name='background')
sample['SiWAFER705V'].material.rho.range(1.9, 2.2)
sample['D2O05V'].material.rho.range(6.2, 6.4)
sample['D2O05V'].interface.range(30.0, 100.0)

################################################################################

expt = Experiment(probe=probe, sample=sample)
problem = FitProblem(expt)
