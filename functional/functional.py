"""
A exemplar Component for an analytic profile.
"""
import numpy as np

from refnx.reflect import ReflectModel, Structure, Component, SLD, Slab
from refnx.analysis import (Bounds, Parameter, Parameters,
                            possibly_create_parameter)


class FunctionalForm(Component):
    """
    Component describing an analytic SLD profile.
    An exponential profile is hard coded here.

    Parameters
    ----------
    extent : Parameter or float
        The total extent of the functional region
    decay : Parameter or float
        Decay length of exponential profile
    rough : Parameter or float
        Roughness between this Component and `left_component`
    left_component : Component
        Prior Component (used to obtain the SLD of the layer immediately
        before this one)
    right_component : Component
        Following Component (used to obtain the SLD of the layer immediately
        after this one)
    name : str
        Name of component
    microslab_max_thickness : float, optional
        Thickness of microslicing of spline for reflectivity calculation

    """

    def __init__(self, extent, decay, rough, left_component, right_component,
                 name='', microslab_max_thickness=1):

        self.name = name

        self.left_component = left_component
        self.right_component = right_component

        self.microslab_max_thickness = microslab_max_thickness

        self.extent = (
            possibly_create_parameter(extent,
                                      name='%s - functional extent' % name))
        self.decay = (
            possibly_create_parameter(decay, name='%s - decay length' % name))

        self.rough = (
            possibly_create_parameter(rough, name='%s - rough' % name))

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.extent, self.decay, self.rough])
        return p

    def lnprob(self):
        return 0

    @property
    def slabs(self):
        num_slabs = np.ceil(float(self.extent) / self.microslab_max_thickness)
        slab_thick = float(self.extent / num_slabs)
        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = slab_thick

        # give last slab a miniscule roughness so it doesn't get contracted
        slabs[-1:, 3] = 0.5

        a = self.left_component.slabs[-1, 1]
        b = self.right_component.slabs[0, 1]

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick

        if a >= b:
            slabs[:, 1] = np.abs(b - a) * np.exp(- dist / self.decay) + a
        else:
            slabs[:, 1] = np.abs(b - a) * (1 - np.exp(- dist / self.decay)) + a

        return slabs
