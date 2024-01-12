
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
SiWAFER70V = SLD(name='SiWAFER70V', rho=2.197, irho=0.0)
D2O = SLD(name='D2O', rho=6.36, irho=0.0)
SiO2 = SLD(name='SiO2', rho=3.001, irho=0.0)
Pd = SLD(name='Pd', rho=3.987, irho=0.0)
void = SLD(name='void', rho=5.797, irho=0.0)
material1intSLD3 = SLD(name='material1intSLD3', rho=4.28, irho=0.0)
material2 = SLD(name='material2', rho=2.588, irho=0.0)
material3 = SLD(name='material3', rho=1.0, irho=0.0)
material4 = SLD(name='material4', rho=2.819, irho=0.0)
material5 = SLD(name='material5', rho=4.441, irho=0.0)
material6 = SLD(name='material6', rho=3.846, irho=0.0)
material7 = SLD(name='material7', rho=3.217, irho=0.0)
material8 = SLD(name='material8', rho=5.145, irho=0.0)


# Film definition ##############################################################
sample = (  D2O(0, 69.61) | material8(148.0, 49.79) | material7(197.9, 49.16) | material6(118.9, 42.1) | material5(126.9, 40.53) | material4(160.9, 38.96) | material3(277.7, 36.93) | material2(117.8, 25.9) | material1intSLD3(92.39, 6.755) | void(18.7, 2.889) | Pd(109.8, 2.065) | SiO2(25.0, 5.977) | SiWAFER70V )

sample['SiO2'].thickness.range(25.0, 30.0)
sample['SiO2'].material.rho.range(3.0, 3.4)
sample['SiO2'].interface.range(2.0, 6.0)
sample['Pd'].thickness.range(100.0, 110.0)
sample['Pd'].material.rho.range(3.9, 4.1)
sample['Pd'].interface.range(2.0, 6.0)
sample['void'].thickness.range(5.0, 100.0)
sample['void'].material.rho.range(1.0, 6.1)
sample['void'].interface.range(1.0, 5.0)
sample['material1intSLD3'].thickness.range(10.0, 500.0)
sample['material1intSLD3'].material.rho.range(1.0, 6.36)
sample['material1intSLD3'].interface.range(2.0, 50.0)
sample['material2'].thickness.range(10.0, 500.0)
sample['material2'].material.rho.range(1.0, 6.36)
sample['material2'].interface.range(10.0, 50.0)
sample['material3'].thickness.range(10.0, 500.0)
sample['material3'].material.rho.range(1.0, 6.36)
sample['material3'].interface.range(10.0, 50.0)
sample['material4'].thickness.range(10.0, 500.0)
sample['material4'].material.rho.range(1.0, 6.36)
sample['material4'].interface.range(10.0, 50.0)
sample['material5'].thickness.range(10.0, 500.0)
sample['material5'].material.rho.range(1.0, 6.36)
sample['material5'].interface.range(10.0, 50.0)
sample['material6'].thickness.range(10.0, 500.0)
sample['material6'].material.rho.range(1.0, 6.36)
sample['material6'].interface.range(10.0, 50.0)
sample['material7'].thickness.range(10.0, 500.0)
sample['material7'].material.rho.range(1.0, 6.36)
sample['material7'].interface.range(10.0, 50.0)
sample['material8'].thickness.range(10.0, 500.0)
sample['material8'].material.rho.range(1.0, 6.36)
sample['material8'].interface.range(10.0, 50.0)



probe.intensity=Parameter(value=1.0,name='normalization')
probe.background=Parameter(value=0.0,name='background')
sample['SiWAFER70V'].material.rho.range(1.9, 2.2)
sample['D2O'].material.rho.range(6.2, 6.36)
sample['D2O'].interface.range(10.0, 70.0)

################################################################################

expt = Experiment(probe=probe, sample=sample)
problem = FitProblem(expt)
