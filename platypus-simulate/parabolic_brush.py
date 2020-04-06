"""
A Component for a brush with a parabolic profile
"""
__author__ = 'Andrew Nelson'
__copyright__ = "Copyright 2019, Andrew Nelson"
__license__ = "3 clause BSD"

import numpy as np
from scipy.special import hyp2f1
from scipy import integrate
from scipy import stats

from refnx.reflect import Component, SLD, Slab
from refnx.reflect.reflect_model import gauss_legendre
from refnx.analysis import (Parameter, Parameters,
                            possibly_create_parameter)


class ParabolicBrush(Component):
    """
    Component describing a parabolic brush (for neutron reflection). It
    should be the last Component before an infinite backing medium.

    Parameters
    ----------
    polymer_sld: SLD
        SLD of the brush
    phi_0: Parameter or float
        initial volume fraction of polymer
    gamma: Parameter or float
        Total interfacial volume of polymer
    alpha: Parameter or float
        exponent of parabolic profile
    delta: Parameter or float
        delta for smearing end of parabolic profile
    rough: Parameter or float
        Roughness between this Component and `left_component`
    name: str
        Name of component
    microslab_max_thickness : float, optional
        Thickness of microslicing of spline for reflectivity calculation

    Notes
    -----
    The characteristic height of the brush is given by `ParabolicBrush.H`. It
    is not a fitting parameter, but is calculated from the unit interval and
    the total interfacial volume.

    Karim, A.; Satija, S. K.; Douglas, J. F.; Ankner, J. F.; Fetters, L. J.
    "Neutron Reflectivity Study of the Density Profile of a Model End- Grafted
     Polymer Brush: Influence of Solvent Quality",
    Phys. Rev. Lett. 1994, 73, 3407−3410.
    """
    def __init__(self, polymer_sld, phi_0, gamma, alpha, delta, rough,
                 name='', microslab_max_thickness=1):
        super(ParabolicBrush, self).__init__(name=name)

        self.polymer_sld = SLD(polymer_sld)
        self.gamma = possibly_create_parameter(gamma,
                                               name='%s - gamma' % name)
        self.microslab_max_thickness = microslab_max_thickness

        self.alpha = (
            possibly_create_parameter(alpha, name='%s - alpha' % name))
        self.delta = (
            possibly_create_parameter(delta, name='%s - delta' % name))
        self.phi_0 = (
            possibly_create_parameter(phi_0, name='%s - phi_0' % name))
        self.rough = (
            possibly_create_parameter(rough, name='%s - rough' % name))

        # self.volume_fraction = np.vectorize(self._volume_fraction)

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.polymer_sld.parameters, self.phi_0, self.gamma,
                  self.alpha, self.delta, self.rough])
        return p

    def volume_fraction(self, z):
        z = np.asfarray(z)
        output = [self._volume_fraction(zed) for zed in z]
        return np.array(output)

    def _volume_fraction(self, z):
        """
        The volume fraction of the brush at distance `z`.

        Parameters:
        -----------
        z: float
            distance
        """
        phi_0, alpha, delta = (self.phi_0.value,
                              self.alpha.value,
                              self.delta.value)

        H = self.H
        sd = delta / (2 * np.sqrt(2 * np.log(2.0)))

        def kernel(x, z0, sd):
            vfp = phi_0 * (1 - (x/H) ** 2) ** alpha
            vfp[x >= H] = 0.
            g = stats.norm.pdf(x, loc=z0, scale=sd)
            return vfp * g

        # might need to convolve with a Gaussian at end of profile
        if (H - 2*delta) <= z <= (H + 2*delta):
            return integrate.fixed_quad(kernel,
                                  z - 3*sd,
                                  z + 3*sd,
                                  args=(z, sd), n=101)[0]
        else:
            return phi_0 * (1 - (z/H) ** 2) ** alpha

    @property
    def H(self):
        """
        Characteristic height of the brush
        """
        required_area = float(self.gamma)
        phi_0, alpha = self.phi_0.value, self.alpha.value

        # Wolfram MathWorld integral
        unscaled_area = phi_0 * hyp2f1(0.5, -alpha, 1.5, 1)

        return required_area / unscaled_area

    def slabs(self, structure=None):
        H = self.H
        delta = self.delta.value
        total_thick = (H + 2*delta)
        num_slabs = np.ceil(total_thick / self.microslab_max_thickness)

        slab_thick = total_thick / num_slabs
        slabs = np.zeros((int(num_slabs), 5))

        slabs[:, 0] = slab_thick
        slabs[:, 1] = self.polymer_sld.real.value
        slabs[:, 2] = self.polymer_sld.imag.value
        slabs[0, 3] = self.rough.value

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick
        vfp = self.volume_fraction(dist)

        slabs[:, 4] = 1 - vfp

        return slabs
