import numpy as np
from scipy.integrate import simpson
from refnx.reflect import Component, SLD, Slab, Structure
from refnx.analysis import possibly_create_parameter, Parameter, Parameters


class MaxEntVFP(Component):
    """
    Uses maximum entropy method to create smooth volume fraction profile (VFP).
    The profile is described by a series of pixels in which the VF is
    allowed to vary.

    Parameters
    ----------
    adsorbed_amount : Parameter
        The total area under the VFP for the dry polymer brush. You should not
        allow this Parameter to vary, but you should set its Bounds to
        the correct prior.

    polymer_sld : Scatterer
        Polymer SLD

    extent : float.
        The total extent of the polymer brush component.

    mj : float
        An estimate of the average volume fraction in the layer. Departing from
        this value adds to the entropic penalty.

    alpha : Parameter
        This value scales the entropy term, `logp = entropy / alpha`. Smaller
        values of alpha will lead to smoother solutions.
        It is suggested that the prior for `alpha` be drawn from a distribution
        that is more probable at smaller positive values (e.g. 1/x).

        .. code-block:: language

            dist = beta(0.7, 500)
            alpha = Parameter(0.01, 'alpha', vary=True)
            alpha.bounds = PDF(dist)

        The rationale is that the fit will want to select smaller values for
        alpha, to make its prior term more probable. This has the effect of
        encouraging smoother VFP solutions. In turn too flat a solution (i.e.
        too similar to an average `mj`) is discouraged by the log-likelihood
        becoming less probable.

    left_slabs : sequence of Slab
    right_slabs
        Slabs that also contribute to the VFP. They are included in the area
        under the VFP. Each Slab should have an SLD described by `polymer.

    microslab_max_thickness : float
        The maximal size of each pixel.

    monotonic : bool
        If the VFP should be described by a monotonic decay. When monotonic is
        True then the absolute VFP in successive pixels is described by:

        .. code-block:: language

            [self.vff[0],
             self.vff[1] * self.vff[0],
              self.vff[2] * self.vff[1] * self.vff[0],
               ...]

        Otherwise the absolute VFP are allowed to vary in [0, 1).

    Attributes
    ----------
    vff : Parameters
        The volume fractions in each pixel (or the multiplier if
        monotonic=True).

    Notes
    -----
    Smooth solutions are encouraged by the use of an entropy penalty that is
    added to logp.
    The number of pixels is chosen as
    ``npixels = int(np.ceil(extent / microslab_max_thickness))``

    References
    ----------
    .. [1] D.S. Sivia, W.A. Hamilton, G.S. Smith,
        Analysis of neutron reflectivity data: maximum entropy, Bayesian spectral
        analysis and speckle holography,
        Physica B: Condensed Matter,
        Volume 173, Issues 1–2,
        1991
    """
    def __init__(self, adsorbed_amount, polymer_sld, extent, mj=0.5, alpha=1.0, name='',
                 left_slabs=(), right_slabs=(),
                 microslab_max_thickness=10.0, monotonic=False):
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

        self.adsorbed_amount = (
            possibly_create_parameter(
                adsorbed_amount,
                name=f'{name} - adsorbed amount'
            )
        )
        self.monotonic = monotonic
        # vf are the volume fraction values of each of the pixels
        self.vff = Parameters(name='vff')
        
        self.alpha = possibly_create_parameter(alpha)
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
                  self.polymer_sld.parameters, self.alpha])
        p.extend([slab.parameters for slab in self.left_slabs])
        p.extend([slab.parameters for slab in self.right_slabs])
        return p

    @property
    def _actual_vfp(self):
        vf = np.array(self.vff)
        if self.monotonic:
            for i in range(1, len(vf)):
                vf[i] = vf[i] * vf[i - 1]
        return vf
    
    def slabs(self, structure=None):
        slabs = np.zeros((self.npixels, 5))
        slabs[:, 0] = self.thickness
        slabs[:, 1] = self.polymer_sld.real.value
        slabs[:, 2] = self.polymer_sld.imag.value
        # no roughness, so can contract
        # vf in each pixel
        vf = np.array(self.vff)
        slabs[:, 4] = 1.0 - self._actual_vfp
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
            
        vf = self._actual_vfp
        area += np.sum(vf * self.thickness)
        return area
        
    def logp(self):
        logp = self.adsorbed_amount.logp(self.gamma())
        logp += self.S()
        return logp

    def S(self):
        # entropy term to add onto total log probability
        vf = self._actual_vfp
        
        S = np.sum((vf - self.mj) - vf*np.log(vf / self.mj))

        for slab in self.left_slabs:
            _slabs = slab.slabs()
            vf = (1 - _slabs[0, 4])
            S += (vf - self.mj) - vf*np.log(vf / self.mj)

        for slab in self.right_slabs:
            _slabs = slab.slabs()
            vf = (1 - _slabs[0, 4])
            S += (vf - self.mj) - vf*np.log(vf / self.mj)
        return S / float(self.alpha)

    def moment(self, moment=1):
        """
        Calculates the n'th moment of the profile

        Parameters
        ----------
        moment : int
            order of moment to be calculated

        Returns
        -------
        moment : float
            n'th moment
        """
        zed, profile = self.profile()
        profile *= zed**moment
        val = simpson(profile, zed)
        return val / self.gamma()

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


class MaxEntVFP_alternate(Component):
    """
    Uses maximum entropy method to create smooth volume fraction profile (VFP).
    The profile is described by a series of pixels in which the VF is
    allowed to vary.

    Parameters
    ----------
    adsorbed_amount : Parameter
        The total area under the VFP for the dry polymer brush.

    polymer_sld : Scatterer
        Polymer SLD

    mj : float
        An estimate of the average volume fraction in the layer. Departing from
        this value adds to the entropic penalty.

    npixels : int
        The number of pixels used to describe the VFP. This is a non-varying
        value.

    alpha : Parameter
        This value scales the entropy term, `logp = entropy / alpha`. Smaller
        values of alpha will lead to smoother solutions.
        It is suggested that the prior for `alpha` be drawn from a distribution
        that is more probable at smaller positive values (e.g. 1/x).

        .. code-block:: language

            dist = beta(0.7, 500)
            alpha = Parameter(0.01, 'alpha', vary=True)
            alpha.bounds = PDF(dist)

        The rationale is that the fit will want to select smaller values for
        alpha, to make its prior term more probable. This has the effect of
        encouraging smoother VFP solutions. In turn too flat a solution (i.e.
        too similar to an average `mj`) is discouraged by the log-likelihood
        becoming less probable.

    left_slabs : sequence of Slab
    right_slabs
        Slabs that also contribute to the VFP. They are included in the area
        under the VFP. Each Slab should have an SLD described by `polymer.

    monotonic : bool
        If the VFP should be described by a monotonic decay. When monotonic is
        True then the absolute VFP in successive pixels is described by:

        .. code-block:: language

            [self.vff[0],
             self.vff[1] * self.vff[0],
              self.vff[2] * self.vff[1] * self.vff[0],
               ...]

        Otherwise the absolute VFP are allowed to vary in [0, 1).

    max_slab_size : float
        `max_slab_size` is used to prevent each pixel growing unreasonably
        large - in the early stages a VFP may be chosen on the unit scale that
        doesn't have much area. The pixel scaling then tries to compensate by
        setting the pixel width to be too large a value (such that the area is
        correct). This can result in the reflectivity calculation failing.
        If the pixels are clipped to a certain size then the fit has
        to compensate by choosing VFs that are reasonable.

    Attributes
    ----------
    vff : Parameters
        The volume fractions in each pixel (or the multiplier if
        monotonic=True).

    Notes
    -----
    Smooth solutions are encouraged by the use of an entropy penalty that is
    added to logp. The maximal extent of this component is
    `npixels * max_slab_size`. This version creates a VFP on a unit
    lengthscale. This unit length scale is then expanded such that the area
    under the VFP is equal to `adsorbed_amount`.

    References
    ----------
    .. [1] D.S. Sivia, W.A. Hamilton, G.S. Smith,
        Analysis of neutron reflectivity data: maximum entropy, Bayesian spectral
        analysis and speckle holography,
        Physica B: Condensed Matter,
        Volume 173, Issues 1–2,
        1991
    """
    def __init__(self, adsorbed_amount, polymer_sld, mj=0.5, npixels=60, alpha=1.0, name='',
                 left_slabs=(), right_slabs=(), monotonic=False, max_slab_size=50):
        super(MaxEntVFP_alternate, self).__init__()
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

        self.adsorbed_amount = (
            possibly_create_parameter(
                adsorbed_amount,
                name=f'{name} - adsorbed amount'
            )
        )
        self.monotonic = monotonic
        self.max_slab_size = max_slab_size
        # vf are the volume fraction values of each of the pixels
        self.vff = Parameters(name='vff')

        self.alpha = possibly_create_parameter(alpha)
        self.mj = mj
        self.npixels = npixels

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
                  self.polymer_sld.parameters, self.alpha])
        p.extend([slab.parameters for slab in self.left_slabs])
        p.extend([slab.parameters for slab in self.right_slabs])
        return p

    @property
    def _actual_vfp(self):
        vf = np.array(self.vff)
        if self.monotonic:
            for i in range(1, len(vf)):
                vf[i] = vf[i] * vf[i - 1]
        return vf

    def slabs(self, structure=None):
        slabs = np.zeros((self.npixels, 5))
        slabs[:, 0] = self._pixel_width()
        slabs[:, 1] = self.polymer_sld.real.value
        slabs[:, 2] = self.polymer_sld.imag.value
        # no roughness, so can contract
        # vf in each pixel
        vf = np.array(self.vff)
        slabs[:, 4] = 1.0 - self._actual_vfp
        return slabs

    def _pixel_width(self):
        required_area = float(self.adsorbed_amount)
        area = 0

        for slab in self.left_slabs:
            _slabs = slab.slabs()
            area += _slabs[0, 0] * (1 - _slabs[0, 4])
        for slab in self.right_slabs:
            _slabs = slab.slabs()
            area += _slabs[0, 0] * (1 - _slabs[0, 4])

        vf = self._actual_vfp
        width_unit_range = 1 / self.npixels
        area_unit_range = np.sum(vf * width_unit_range)
        scaler = (required_area - area) / area_unit_range

        return np.clip(scaler * width_unit_range, 0.1, self.max_slab_size)

    def gamma(self):
        # calculates adsorbed amount
        area = 0

        for slab in self.left_slabs:
            _slabs = slab.slabs()
            area += _slabs[0, 0] * (1 - _slabs[0, 4])
        for slab in self.right_slabs:
            _slabs = slab.slabs()
            area += _slabs[0, 0] * (1 - _slabs[0, 4])

        vf = self._actual_vfp
        pixel_width = self._pixel_width()

        area += np.sum(vf * pixel_width)

        return area

    def logp(self):
        logp = self.adsorbed_amount.logp(self.gamma())
        logp += self.S()
        return logp

    def S(self):
        # entropy term to add onto total log probability
        vf = self._actual_vfp

        S = np.sum((vf - self.mj) - vf * np.log(vf / self.mj))

        for slab in self.left_slabs:
            _slabs = slab.slabs()
            vf = (1 - _slabs[0, 4])
            S += (vf - self.mj) - vf * np.log(vf / self.mj)

        for slab in self.right_slabs:
            _slabs = slab.slabs()
            vf = (1 - _slabs[0, 4])
            S += (vf - self.mj) - vf * np.log(vf / self.mj)
        return S / float(self.alpha)

    def moment(self, moment=1):
        """
        Calculates the n'th moment of the profile

        Parameters
        ----------
        moment : int
            order of moment to be calculated

        Returns
        -------
        moment : float
            n'th moment
        """
        zed, profile = self.profile()
        profile *= zed**moment
        val = simpson(profile, zed)
        return val / self.gamma()

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
