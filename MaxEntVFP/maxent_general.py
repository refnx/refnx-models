import numpy as np
from scipy.integrate import simpson
from refnx.reflect import Component, SLD, Slab, Structure
from refnx.analysis import possibly_create_parameter, Parameter, Parameters


class MaxEnt(Component):
    """
    Uses maximum entropy method for free form modelling.
    The profile over which this model operates is described by a series of
    pixels in which the SLD is allowed to freely vary.

    Parameters
    ----------
    npixels : int
        Number of pixels used in definining the MaxEnt region.

    slab_thickness : Parameter
        Width of each pixel. Should be positive.

    low_SLD : float
        Minimum value of pixel SLD.

    high_SLD : float
        Maximum value of pixel SLD.

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

    Attributes
    ----------
    vff : Parameters
        The SLD in each pixel.

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

    def __init__(
        self, npixels, slab_thickness, low_SLD, high_SLD, mj, alpha=1.0, name=""
    ):
        super(MaxEnt, self).__init__()
        self.name = name

        self.npixels = int(npixels)
        self.slab_thickness = possibly_create_parameter(slab_thickness)

        # vff are the SLD values of each of the pixels
        self.vff = Parameters(name="vff")

        self.alpha = possibly_create_parameter(alpha)
        self.mj = mj
        self.low_SLD = low_SLD
        self.high_SLD = high_SLD
        rng = np.random.default_rng()

        for i in range(self.npixels):
            p = Parameter(
                rng.uniform(low_SLD, high_SLD), name=f"{name} - pixel vff{i}"
            )
            p.range(low_SLD, high_SLD)
            p.vary = True
            self.vff.append(p)

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.vff, self.slab_thickness, self.alpha])
        return p

    @property
    def _actual_sld(self):
        vf = np.array(self.vff)
        return vf

    def slabs(self, structure=None):
        slds = self._actual_sld

        slabs = np.zeros((self.npixels, 5))
        slabs[:, 0] = float(self.slab_thickness)
        slabs[:, 1] = slds
        # no roughness, so can contract sld in each pixel
        return slabs

    def logp(self):
        logp = self.S()
        return logp

    def S(self):
        # entropy term to add onto total log probability
        vf = self._actual_sld

        S = np.sum((vf - self.mj) - vf * np.log(vf / self.mj))

        return S / float(self.alpha)
