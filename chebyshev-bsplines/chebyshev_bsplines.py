import numpy as np
from refnx.reflect import Component
from refnx.analysis import possibly_create_parameter, Parameters, Parameter
from numpy.polynomial.chebyshev import Chebyshev, chebpts1
from scipy.interpolate import make_interp_spline


class Cheby(Component):
    _methods = ['direct', 'interp']

    def __init__(self, thick, vs, method='direct', order=None, dzf=None, name=None, microslab_max_thickness=1):
        self.name = name
        self._interfaces = None

        self.thick = possibly_create_parameter(thick)
        self.method = method
        if method not in ['direct', 'interp']:
            raise ValueError("method must either be 'direct' or 'interp'")

        if isinstance(vs, Parameters):
            self.vs = vs
        else:
            self.vs = Parameters([possibly_create_parameter(v) for v in vs])

        if isinstance(dzf, Parameters):
            self.dzf = dzf
        else:
            if dzf is not None and method == 'interp':
                self.dzf = Parameters([possibly_create_parameter(v) for v in dzf])
                if len(dzf) != len(self.vs) + 1:
                    raise ValueError("len(dzf) must be one larger than len(vs)")
            else:
                self.dzf = None

        try:
            if method == 'interp':
                self.order = int(order)

        except TypeError:
            raise TypeError("You must specify a positive integer for 'order' when you use method='interp'")

        self.microslab_max_thickness = microslab_max_thickness

    def _zeds(self):
        if self.dzf is None:
            return self.thick.value * 0.5 * (chebpts1(len(self.vs)) + 1.0)

        zeds = np.cumsum(self.dzf)
        # Normalise dzf to unit interval.
        # clipped to 0 and 1 because we pad on the LHS, RHS later
        # and we need the array to be monotonically increasing
        zeds /= zeds[-1]
        zeds = np.clip(zeds, 0, 1)
        zeds = zeds[0:-1]
        return zeds * self.thick.value

    def slabs(self, structure=None):
        nslabs = int(np.ceil(self.thick.value / self.microslab_max_thickness))

        if self.method == "direct":
            cheb = Chebyshev(np.array(self.vs), domain=(0, self.thick.value))
        else:
            zeds = self._zeds()
            cheb = Chebyshev.fit(zeds, np.array(self.vs), domain=(0, self.thick.value), deg=self.order)

        slab_thick = self.thick.value / nslabs
        zed = slab_thick * (np.arange(nslabs) + 0.5)
        rho = cheb(zed)
        slabs = np.zeros((nslabs, 5), float)
        slabs[:, 0] = slab_thick
        slabs[:, 1] = rho
        return slabs

    @property
    def parameters(self):
        _p = Parameters([self.thick])
        _p.extend(self.vs)
        if self.dzf is not None:
            _p.extend(self.dzf)
        return _p


class CubicBSpline(Component):
    _methods = ['direct', 'interp']

    def __init__(self, thick, vs, dzf=None, name=None, microslab_max_thickness=1):
        self.name = name
        self._interfaces = None

        self.thick = possibly_create_parameter(thick)

        if isinstance(vs, Parameters):
            self.vs = vs
        else:
            self.vs = Parameters([possibly_create_parameter(v) for v in vs])

        if isinstance(dzf, Parameters):
            self.dzf = dzf
        else:
            if dzf is not None:
                self.dzf = Parameters([possibly_create_parameter(v) for v in dzf])
                if len(dzf) != len(self.vs) + 1:
                    raise ValueError("len(dzf) must be one larger than len(vs)")
            else:
                self.dzf = None

        self.microslab_max_thickness = microslab_max_thickness

    def _zeds(self):
        if self.dzf is None:
            npnts = len(self.vs)
            return self.thick.value / npnts * (np.arange(npnts) + 0.5)

        zeds = np.cumsum(self.dzf)
        # Normalise dzf to unit interval.
        # clipped to 0 and 1 because we pad on the LHS, RHS later
        # and we need the array to be monotonically increasing
        zeds /= zeds[-1]
        zeds = np.clip(zeds, 0, 1)
        zeds = zeds[0:-1]
        return zeds * self.thick.value

    def slabs(self, structure=None):
        nslabs = int(np.ceil(self.thick.value / self.microslab_max_thickness))
        zeds = self._zeds()

        bspl = make_interp_spline(zeds, np.array(self.vs))

        slab_thick = self.thick.value / nslabs
        zed = slab_thick * (np.arange(nslabs) + 0.5)
        rho = bspl(zed)
        slabs = np.zeros((nslabs, 5), float)
        slabs[:, 0] = slab_thick
        slabs[:, 1] = rho
        return slabs

    @property
    def parameters(self):
        _p = Parameters([self.thick])
        _p.extend(self.vs)
        if self.dzf is not None:
            _p.extend(self.dzf)
        return _p
