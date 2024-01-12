
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
air = SLD(name='air', rho=0.0, irho=0.0)
SiWAFER7 = SLD(name='SiWAFER7', rho=2.074, irho=0.0)
polymerNIPAMFC = SLD(name='polymerNIPAMFC', rho=0.9235, irho=0.0)
Pd = SLD(name='Pd', rho=4.02, irho=0.0)
SiO2 = SLD(name='SiO2', rho=3.185, irho=0.0)


# Film definition ##############################################################
sample = (  SiWAFER7(0, 4.973) | SiO2(28.46, 3.596) | Pd(106.6, 6.746) | polymerNIPAMFC(585.4, 3.845) | air )

sample['polymerNIPAMFC'].thickness.range(300.0, 800.0)
sample['polymerNIPAMFC'].material.rho.range(0.0, 4.0)
sample['polymerNIPAMFC'].interface.range(0.1, 10.0)
sample['Pd'].thickness.range(50.0, 200.0)
sample['Pd'].material.rho.range(3.6, 4.2)
sample['Pd'].interface.range(1.0, 10.0)
sample['SiO2'].thickness.range(1.0, 50.0)
sample['SiO2'].material.rho.range(2.0, 4.5)
sample['SiO2'].interface.range(1.0, 10.0)



probe.intensity.range(0.9, 1.1)
probe.background.range(0.0, 6e-07)
sample['SiWAFER7'].material.rho.range(1.7, 2.3)
sample['SiWAFER7'].interface.range(0.0, 10.0)

################################################################################

expt = Experiment(probe=probe, sample=sample)
problem = FitProblem(expt)
