"""Use Maximum Entropy methods for volume fraction profiles"""
"""
Sivia et al., "Analysis of neutron reflectivity data: maximum entropy,
Bayesian Spectral Analysis and speckle holography", Physica B, 173 (1991),
121-138
"""
import numpy as np

from refnx.analysis import (
    Parameter,
    possibly_create_parameter,
    Parameters,
)
from refnx.reflect import Component, Slab, SLD, Structure


class MaxEntVFP(Component):
    def __init__(self, adsorbed_amount, polymer_sld, extent, mj=0.5, alpha=1.0, name='',
                 left_slabs=(), right_slabs=(),
                 microslab_max_thickness=10.0):
        """
        Parameters
        ----------
        adsorbed_amount : float, Parameter
            Total volume of polymer in the profile.
        polymer_sld : SLD, float, complex
            SLD of polymer in the profile.
        extent : float, Parameter
            Total extent of the profile.
        mj : Parameter, float
            Average volume fraction expected in the layer, [0, 1]
        alpha : float
            regularising factor.
        name : str
            Name of component
        left_slabs : sequence of Slab
            Polymer Slabs to the left of the spline
        right_slabs : sequence of Slab
            Polymer Slabs to the right of the spline
        microslab_max_thickness : float
            Thickness of microslicing of spline for reflectivity calculation.

        Notes
        -----
        THIS IS EXPERIMENTAL CODE.
        """
        super(MaxEntVFP, self).__init__()
        self.name = name

        if isinstance(polymer_sld, SLD):
            self.polymer_sld = polymer_sld
        else:
            self.polymer_sld = SLD(polymer_sld)

        # left and right slabs are other areas where the same polymer can
        # reside
        self.left_slabs = [slab for slab in left_slabs if
                           isinstance(slab, Slab)]
        self.right_slabs = [slab for slab in right_slabs if
                            isinstance(slab, Slab)]

        if len(self.left_slabs):
            self.start_vf = 1 - self.left_slabs[-1].vfsolv.value
        else:
            self.start_vf = 1

        # in contrast use a vf = 0 for the last vf of
        # the spline, unless right_slabs is specified
        if len(self.right_slabs):
            self.end_vf = 1 - self.right_slabs[0].vfsolv.value
        else:
            self.end_vf = 1e-15

        self.adsorbed_amount = (
            possibly_create_parameter(
                adsorbed_amount,
                name=f'{name} - adsorbed amount'
            )
        )

        # vf are the volume fraction values of each of the pixels
        self.vff = Parameters(name='vff')

        self.alpha = alpha
        self.extent = extent
        self.mj = mj
        self.npixels = int(np.ceil(extent / microslab_max_thickness))
        self.thickness = extent / self.npixels

        for i in range(self.npixels):
            p = Parameter(
                mj,
                name=f'{name} - pixel vff{i}')
            p.range(1.0e-15, 1.0)
            self.vff.append(p)

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.adsorbed_amount, self.vff,
                  self.polymer_sld.parameters])
        p.extend([slab.parameters for slab in self.left_slabs])
        p.extend([slab.parameters for slab in self.right_slabs])
        return p

    def slabs(self, structure=None):
        slabs = np.zeros((self.npixels, 5))
        slabs[:, 0] = self.thickness
        slabs[:, 1] = self.polymer_sld.real.value
        slabs[:, 2] = self.polymer_sld.imag.value
        # no roughness, so can contract
        # vf in each pixel
        vf = np.array(self.vff)
        slabs[:, 4] = 1.0 - vf
        return slabs

    def gamma(self):
        # calculates adsorbed amount
        area = 0

        for slab in self.left_slabs:
            _slabs = slab.slabs()
            area += _slabs[0, 0] * (1 - _slabs[0, 4])
        for slab in self.right_slabs:
            _slabs = slab.slabs()
            area += _slabs[0, 0] * (1 - _slabs[0, 4])

        vf = np.array(self.vff)
        area += np.sum(vf * self.thickness)
        return area

    def logp(self):
        logp = self.adsorbed_amount.logp(self.gamma())
        logp += self.S()
        return logp

    def S(self):
        vf = np.array(self.vff)
        S = np.sum((vf - self.mj) - vf * np.log(vf / self.mj))

        for slab in self.left_slabs:
            _slabs = slab.slabs()
            vf = (1 - _slabs[0, 4])
            S += (vf - self.mj) - vf * np.log(vf / self.mj)

        for slab in self.right_slabs:
            _slabs = slab.slabs()
            vf = (1 - _slabs[0, 4])
            S += (vf - self.mj) - vf * np.log(vf / self.mj)
        return self.alpha * S

    def profile(self, extra=False):
        """
        Calculates the volume fraction profile

        Returns
        -------
        z, vfp : np.ndarray
            Distance from the interface, volume fraction profile
        """
        s = Structure()
        s |= SLD(0)

        m = SLD(1.)

        for i, slab in enumerate(self.left_slabs):
            layer = m(slab.thick.value, slab.rough.value)
            if not i:
                layer.rough.value = 0
            layer.vfsolv.value = slab.vfsolv.value
            s |= layer

        polymer_slabs = self.slabs()
        offset = np.sum(s.slabs()[:, 0])

        for i in range(np.size(polymer_slabs, 0)):
            layer = m(polymer_slabs[i, 0], polymer_slabs[i, 3])
            layer.vfsolv.value = polymer_slabs[i, -1]
            s |= layer

        for i, slab in enumerate(self.right_slabs):
            layer = m(slab.thick.value, slab.rough.value)
            layer.vfsolv.value = slab.vfsolv.value
            s |= layer

        s |= SLD(0, 0)

        # now calculate the VFP.
        total_thickness = np.sum(s.slabs()[:, 0])
        if total_thickness < 500:
            num_zed_points = total_thickness
        else:
            num_zed_points = 500
        zed = np.linspace(0, total_thickness, num_zed_points)
        # SLD profile puts a very small roughness on the interfaces with zero
        # roughness.
        zed[0] = 0.01
        z, s = s.sld_profile(z=zed)
        s[0] = s[1]
        return z, s
