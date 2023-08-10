## This code was adapted from that of Paul Kienzle (NIST) and Richard Sheridan.
## The original code was public domain.

## The original code was designed for use with Refl1D.
## Isaac Gresham modified it for compatability with refnx.

## Further optimisations are required for nSCFT_VFP to be
## compatible with 

from collections import OrderedDict
import numpy as np
from numpy.core.multiarray import correlate as old_correlate

from scipy.interpolate import PchipInterpolator as Pchip
from scipy.integrate import simps

from refnx.reflect import Structure, Component, SLD, Slab
from refnx.analysis import Parameters, Parameter, possibly_create_parameter

LAMBDA_1 = 1.0/6.0 #always assume cubic lattice (1/6) for now
LAMBDA_0 = 1.0 - 2.0*LAMBDA_1
# Use reverse order for LAMBDA_ARRAY if it is asymmetric since we are using
# it with correlate().
LAMBDA_ARRAY = np.array([LAMBDA_1, LAMBDA_0, LAMBDA_1])
MINLAT = 25
MINBULK = 5
SQRT_PI = np.sqrt(np.pi)

_SZdist_dict = OrderedDict()



EPS = np.finfo(float).eps


class nSCFT_VFP(Component):
    def __init__(self, adsorbed_amount, lattice_size, polymerMW, latticeMW, chi, chi_s, pdi, polymer_sld, rough=1, name='',
                 microslab_max_thickness=1, dist_model=None, clear_cache=False):
        """
        Parameters
        ----------
        Adsorbed Amount : Parameter or float
            The integral of the volume fraction profile, equivalent to the dry thickness

        polymer_sld : SLD object or float
            The scattering length density of the polymer component
            
        chi : float
            Flory-Huggins solvent interaction parameter

        chi_s :
            surface interaction parameter
            
        l_lat :
            real length per lattice site
            
        mn : float
            Number average molecular weight
        
        m_lat : float
            real mass per lattice segment
            
        pdi : float
            Dispersity (Polydispersity index). float >= 1. By default,
            calculated using the schulz-zimm distribution
            
        name : string
            The name of the structural component. Useful for later introspeciton.
            
        microslab_max_thickness : float
            Maximum slab thickness when approximating the profile as a series of slabs.
    
        rough : float
            the (gaussian) roughness between slabs in the slab approximation of the profile
            
        dist_model : function or None
            The model used to turn the PDI paramter into a probability density function of chain lengths.
            If none the SZdist model is used.
            
        clear_cache : bool
            If true, performs every nSCFT calculation from scratch. If false, uses the prior (cached)
            position as the starting point of the simulation. When using gradient decent fitters, 
            settering clear_cache to False speeds up the process considerably. When Using evolution
            based fitters, monte carlo methods, or manualy tuning, clear_cache should be set to True.
            
        """
        super(nSCFT_VFP, self).__init__()
        
        self.clear_cache=clear_cache

        self.name = name

        if isinstance(polymer_sld, SLD):
            self.polymer_sld = polymer_sld
        else:
            self.polymer_sld = SLD(polymer_sld)


        if dist_model is None:
            print ('no dist model provided, using schulz-zimm distribution')
            dist_model = SZdist

        self.dist_model = dist_model

        self.microslab_max_thickness = microslab_max_thickness

        self.adsorbed_amount = (
            possibly_create_parameter(adsorbed_amount,
                                      name='%s - adsorbed amount' % name))

        self.lattice_size = (
            possibly_create_parameter(lattice_size,
                                      name='%s - lattice size' % name))
        
        self.polymerMW = (
            possibly_create_parameter(polymerMW,
                                      name='%s - polymer MW' % name))
        
        self.latticeMW = (
            possibly_create_parameter(latticeMW,
                                      name='%s - lattice MW' % name))
        
        self.chi = (
            possibly_create_parameter(chi,
                                      name='%s - chi' % name))
        
        self.chi_s = (
            possibly_create_parameter(chi_s,
                                      name='%s - chi substrate' % name))
        
        self.pdi = (
            possibly_create_parameter(pdi,
                                      name='%s - polydispersity index' % name))

        self.rough = (
            possibly_create_parameter(rough,
                                      name='%s - roughness' % name))



    def __call__(self, z):
        """
        Calculates the volume fraction profile of the spline

        Parameters
        ----------
        z : float
            Distance along vfp

        Returns
        -------
        vfp : float
            Volume fraction
        """
        vfp = SCFprofile(z, chi=self.chi.value, chi_s=self.chi_s.value, h_dry=self.adsorbed_amount.value,
                         l_lat=self.lattice_size.value, mn=self.polymerMW.value, m_lat=self.latticeMW.value,
                         pdi=self.pdi.value, dist_model=self.dist_model, clear_cache=self.clear_cache)
        return vfp

    

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
        val = simps(profile, zed)
        area = self.profile_area()
        return val / area

    def is_monotonic(self):
        return np.all(self.dzf.pvals < 1)

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.adsorbed_amount, self.chi, self.chi_s,
                  self.polymer_sld.parameters, self.lattice_size,
                  self.polymerMW, self.latticeMW, self.pdi, self.rough])
        return p

    def lnprob(self):
        return 0

    def profile_area(self, bounds=[0,2000]):
        """
        Calculates integrated area of volume fraction profile

        Returns
        -------
        area: integrated area of volume fraction profile
        """
        
        z = np.linspace(*bounds, 10000)

        return np.trapz(self(z), z)

    def slabs(self, structure=None):
        
        slab_extent = self.lattice_size.value*self.polymerMW.value/self.latticeMW.value
        num_slabs = np.ceil(float(slab_extent) / self.microslab_max_thickness)
        slab_thick = float(slab_extent / num_slabs)
        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = slab_thick

        # give last slab a miniscule roughness so it doesn't get contracted
        slabs[-1:, 3] = 0.5

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick
        

        slabs[:, 1] = self.polymer_sld.real.value
        slabs[:, 2] = self.polymer_sld.imag.value
        slabs[:, 4] = 1 - self(dist)
        
        slabs[0:, 3] = self.rough.value

        return slabs

    def profile(self):
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
            layer.vfsolv.value = 1 - slab.vfsolv.value
            s |= layer

        s |= SLD(0, 0)

        # now calculate the VFP.
        total_thickness = np.sum(s.slabs()[:, 0])
        if total_thickness < 500:
            num_zed_points = int(total_thickness)
        else:
            num_zed_points = 500
        zed = np.linspace(0, total_thickness, num_zed_points)
        # SLD profile puts a very small roughness on the interfaces with zero
        # roughness.
        zed[0] = 0.01
        z, s = s.sld_profile(z=zed)
        s[0] = s[1]


        return z, s

def SZdist(pdi, nn, cache=_SZdist_dict):
    """ Calculate Shultz-Zimm distribution from PDI and number average DP

    Shultz-Zimm is a "realistic" distribution for linear polymers. Numerical
    problems arise when the distribution gets too uniform, so if we find them,
    default to an exact uniform calculation.
    """
    from scipy.special import gammaln
    args = pdi, nn
    if args in cache:
        cache[args] = cache.pop(args)
        return cache[args]

    uniform = False

    if pdi == 1.0:
        uniform = True
    elif pdi < 1.0:
        raise ValueError('Invalid PDI')
    else:
        x = 1.0/(pdi-1.0)
        # Calculate the distribution in chunks so we don't waste CPU time
        chunk = 256
        p_ni_list = []
        pdi_underflow = False

        for i in range(max(1, int((100*nn)/chunk))):
            ni = np.arange(chunk*i+1, chunk*(i+1)+1, dtype=np.float64)
            r = ni/nn
            xr = x*r

            p_ni = np.exp(np.log(x/ni) - gammaln(x+1) + xr*(np.log(xr)/r-1))

            pdi_underflow = (p_ni>=1.0).any() # catch "too small PDI"
            if pdi_underflow:
                break # and break out to uniform calculation

            # Stop calculating when species account for less than 1ppm
            keep = (r < 1.0) | (p_ni >= 1e-6)
            if keep.all():
                p_ni_list.append(p_ni)
            else:
                p_ni_list.append(p_ni[keep])
                break
        else: # Belongs to the for loop. Executes if no break statement runs.
            raise RuntimeError('SZdist overflow')

    if uniform or pdi_underflow:
        # NOTE: rounding here allows nn to be a double in the rest of the np.logic
        p_ni = np.zeros(int(round(nn)))
        p_ni[-1] = 1.0
    else:
        p_ni = np.concatenate(p_ni_list)
        p_ni /= p_ni.sum()
    cache[args]=p_ni

    if len(cache)>9000:
        cache.popitem(last=False)

    return p_ni


def SCFprofile(z, chi=None, chi_s=None, h_dry=None, l_lat=1, mn=None,
               m_lat=1, phi_b=0, pdi=1, disp=False, dist_model=SZdist, clear_cache=False):
    """
    Polymer end-tethered to an interface in a solvent.
    Generate volume fraction profile for Refl1D based on real parameters.

    The field theory is a lattice-based one, so we need to move between lattice
    and real space. This is done using the parameters l_lat and m_lat, the
    lattice size and the mass of a lattice segment, respectivley. We use h_dry
    (dry thickness) as a convenient measure of surface coverage, along with mn
    (number average molecular weight) as the real inputs.

    Make sure your inputs for h_dry/l_lat and mn/m_lat match dimensions!
    Angstroms and daltons are good choices.

    Uses a numerical self-consistent field profile.\ [#Cosgrove]_\ [#deVos]_\ [#Sheridan]_

    **Parameters**
        *chi*
            solvent interaction parameter
        *chi_s*
            surface interaction parameter
        *h_dry*
            thickness of the neat polymer layer
        *l_lat*
            real length per lattice site
        *mn*
            Number average molecular weight
        *m_lat*
            real mass per lattice segment
        *pdi*
            Dispersity (Polydispersity index)
        *phi_b*
            volume fraction of free chains in solution. useful for associating
            grafted films e.g. PS-COOH in Toluene with an SiO2 surface.


    Previous layer should not have roughness! Use a spline to simulate it.

    According to [#Vincent]_, $l_\text{lat}$ and $m_\text{lat}$ should be
    calculated by the formulas:

    .. math::

        l_\text{lat} &= \frac{a^2 m/l}{p_l} \\
        m_\text{lat} &= \frac{(a m/l)^2}{p_l}

    where $l$ is the real polymer's bond length, $m$ is the real segment mass,
    and $a$ is the ratio between molecular weight and radius of gyration at
    theta conditions. The lattice persistence, $p_l$, is:

    .. math::

        p_l = \frac16 \frac{1+1/Z}{1-1/Z}

    with coordination number $Z = 6$ for a cubic lattice, $p_l = .233$.

    """

    # calculate lattice space parameters
    theta = h_dry/l_lat
    segments = mn/m_lat
    sigma = theta/segments

    # solve the self consistent field equations using the cache
    if disp:
        print("\n=====Begin calculations=====\n")
        
    if clear_cache:
        phi_lat = SCFcache(chi, chi_s, pdi, sigma, phi_b, segments, dist_model, disp, cache=OrderedDict())
    else:
        phi_lat = SCFcache(chi, chi_s, pdi, sigma, phi_b, segments, dist_model, disp)

    if disp:
        print("\n============================\n")

    # Chop edge effects out
    for x, layer in enumerate(reversed(phi_lat)):
        if abs(layer - phi_b) < 1e-6:
            break
    phi_lat = phi_lat[:-(x + 1)]

    # re-dimensionalize the solution
    layers = len(phi_lat)
    z_end = l_lat*layers
    z_lat = np.linspace(0.0, z_end, num=layers)
    phi = np.interp(z, z_lat, phi_lat, right=phi_b)

    return phi

_SCFcache_dict = OrderedDict()


def SCFcache(chi, chi_s, pdi, sigma, phi_b, segments, dist_model, disp=False,
             cache=_SCFcache_dict):
    """Return a memoized SCF result by walking from a previous solution.

    Using an OrderedDict because I want to prune keys FIFO
    """
    from scipy.optimize.nonlin import NoConvergence
    # prime the cache with a known easy solutions
    if not cache:
        cache[(0, 0, 0, .1, .1, .1)] = SCFsolve(sigma=.1, phi_b=.1, segments=50, disp=disp)
        cache[(0, 0, 0, 0, .1, .1)] = SCFsolve(sigma=0, phi_b=.1, segments=50, disp=disp)
        cache[(0, 0, 0, .1, 0, .1)] = SCFsolve(sigma=.1, phi_b=0, segments=50, disp=disp)

    if disp:
        starttime = time()

    # Try to keep the parameters between 0 and 1. Factors are arbitrary.
    scaled_parameters = (chi, chi_s * 3, pdi - 1, sigma, phi_b, segments / 500)

    # longshot, but return a cached result if we hit it
    if scaled_parameters in cache:
        if disp:
            print('SCFcache hit at:', scaled_parameters)
        phi = cache[scaled_parameters] = cache.pop(scaled_parameters)
        return phi

    # Find the closest parameters in the cache: O(len(cache))

    # Numpy setup
    cached_parameters = tuple(dict.__iter__(cache))
    cp_array = np.array(cached_parameters)
    p_array = np.array(scaled_parameters)

    # Calculate distances to all cached parameters
    deltas = p_array - cp_array # Parameter space displacement vectors
    closest_index = np.sum(deltas * deltas, axis=1).argmin()

    # Organize closest point data for later use
    closest_cp = cached_parameters[closest_index]
    closest_cp_array = cp_array[closest_index]
    closest_delta = deltas[closest_index]

    phi = cache[closest_cp] = cache.pop(closest_cp)

    if disp:
        print("Walking from nearest:", closest_cp_array)
        print("to:", p_array)

    """
    We must walk from the previously cached point to the desired region.
    This is goes from step=0 (cached) and step=1 (finish), where the step=0
    is implicit above. We try the full step first, so that this function only
    calls SCFsolve one time during normal cache misses.

    The solver may not converge if the step size is too big. In that case,
    we retry with half the step size. This should find the edge of the basin
    of attraction for the solution eventually. On successful steps we increase
    stepsize slightly to accelerate after getting stuck.

    It might seem to make sense to bin parameters into a coarser grid, so we
    would be more likely to have cache hits and use them, but this rarely
    happened in practice.
    """

    step = 1.0 # Fractional distance between cached and requested
    dstep = 1.0 # Step size increment
    flag = True

    while flag:
        # end on 1.0 exactly every time
        if step >= 1.0:
            step = 1.0
            flag = False

        # conditional math because, "why risk floating point error"
        if flag:
            p_tup = tuple(closest_cp_array + step*closest_delta)
        else:
            p_tup = scaled_parameters

        if disp:
            print('Parameter step is', step)
            print('current parameters:', p_tup)

        try:
            phi = SCFsolve(p_tup[0], p_tup[1] / 3, p_tup[2] + 1, p_tup[3], p_tup[4],
                           p_tup[5] * 500, disp=disp, phi0=phi, dist_model=dist_model)
        except (NoConvergence, ValueError) as e:
            if isinstance(e, ValueError):
                if str(e) != "array must not contain infs or NaNs":
                    raise
            if disp:
                print('Step failed')
            flag = True  # Reset this so we don't quit if step=1.0 fails
            dstep *= .5
            step -= dstep
            if dstep < 1e-5:
                raise RuntimeError('Cache walk appears to be stuck')
        else:  # Belongs to try, executes if no exception is raised
            cache[p_tup] = phi
            dstep *= 1.05
            step += dstep

    if disp:
        print('SCFcache execution time:', round(time()-starttime, 3), "s")

    # keep the cache from consuming all things
    while len(cache) > 100:
        cache.popitem(last=False)

    return phi

def SCFsolve(chi=0, chi_s=0, pdi=1, sigma=None, phi_b=0, segments=None,
             disp=False, phi0=None, maxiter=30, dist_model=SZdist):
    """Solve SCF equations using an initial guess and lattice parameters

    This function finds a solution for the equations where the lattice size
    is sufficiently large.

    The Newton-Krylov solver really makes this one. With gmres, it was faster
    than the other solvers by quite a lot.
    """

    from scipy.optimize import newton_krylov

    if sigma >= 1:
        raise ValueError('Chains that short cannot be squeezed that high')

    if disp:
        starttime = time()
        
    # print ('segments: ', segments, 'pdi: ', pdi)

    p_i = dist_model(pdi, segments)
    # print ('len dist: ', len(p_i))

    if phi0 is None:
        # TODO: Better initial guess for chi>.6
        phi0 = default_guess(segments, sigma)
        if disp:
            print('No guess passed, using default phi0: layers =', len(phi0))
    else:
        phi0 = abs(phi0)
        phi0[phi0>.99999] = .99999
        if disp:
            print("Initial guess passed: layers =", len(phi0))

    # resizing loop variables
    jac_solve_method = 'gmres'
    lattice_too_small = True

    # We tolerate up to 1 ppm deviation from bulk phi
    # when counting layers_near_phi_b
    tol = 1e-6

    def curried_SCFeqns(phi):
        return SCFeqns(phi, chi, chi_s, sigma, segments, p_i, phi_b)

    while lattice_too_small:
        if disp:
            print("Solving SCF equations")

        try:
            with np.errstate(invalid='ignore'):
                phi = abs(newton_krylov(curried_SCFeqns,
                                        phi0,
                                        verbose=bool(disp),
                                        maxiter=maxiter,
                                        method=jac_solve_method,
                                        ))
        except RuntimeError as e:
            if str(e) == 'gmres is not re-entrant':
                # Threads are racing to use gmres. Lose the race and use
                # something slower but thread-safe.
                jac_solve_method = 'lgmres'
                continue
            else:
                raise

        if disp:
            print('lattice size:', len(phi))

        phi_deviation = abs(phi - phi_b)
        layers_near_phi_b = phi_deviation < tol
        nbulk = np.sum(layers_near_phi_b)
        lattice_too_small = nbulk < MINBULK

        if lattice_too_small:
            # if there aren't enough layers_near_phi_b, grow the lattice 20%
            newlayers = max(1, round(len(phi0) * 0.2))
            if disp:
                print('Growing undersized lattice by', newlayers)
            if nbulk:
                i = np.diff(layers_near_phi_b).nonzero()[0].max()
            else:
                i = phi_deviation.argmin()
            phi0 = np.insert(phi, i, np.linspace(phi[i - 1], phi[i], num=newlayers))

    if nbulk > 2 * MINBULK:
        chop_end = np.diff(layers_near_phi_b).nonzero()[0].max()
        chop_start = chop_end - MINBULK
        i = np.arange(len(phi))
        phi = phi[(i <= chop_start) | (i > chop_end)]

    if disp:
        print("SCFsolve execution time:", round(time()-starttime, 3), "s")

    return phi





def default_guess(segments=100, sigma=.5, phi_b=.1, chi=0, chi_s=0):
    """ Produce an initial guess for phi via analytical approximants.

    For now, a line using numbers from scaling theory
    """
    ss=np.sqrt(sigma)
    default_layers = int(round(max(MINLAT, segments * ss)))
    default_phi0 = np.linspace(ss, phi_b, num=default_layers)
    return default_phi0


def SCFeqns(phi_z, chi, chi_s, sigma, n_avg, p_i, phi_b=0):
    """ System of SCF equation for terminally attached polymers.

        Formatted for input to a nonlinear minimizer or solver.

        The sign convention here on u is "backwards" and always has been.
        It saves a few sign flips, and looks more like Cosgrove's.
    """

    # let the solver go negative if it wants
    phi_z = abs(phi_z)

    # penalize attempts that overfill the lattice
    toomuch = phi_z > .99999
    penalty_flag = toomuch.any()
    if penalty_flag:
        penalty = np.where(toomuch, phi_z - .99999, 0)
        phi_z[toomuch] = .99999

    # calculate new g_z (Boltzmann weighting factors)
    u_prime = np.log((1.0 - phi_z) / (1.0 - phi_b))
    u_int = 2 * chi * (old_correlate(phi_z, LAMBDA_ARRAY, 1) - phi_b)
    u_int[0] += chi_s
    u_z = u_prime + u_int
    g_z = np.exp(u_z)

    # normalize g_z for numerical stability
    u_z_avg = np.mean(u_z)
    g_z_norm = g_z / np.exp(u_z_avg)

    phi_z_new = calc_phi_z(g_z_norm, n_avg, sigma, phi_b, u_z_avg, p_i)

    eps_z = phi_z - phi_z_new

    if penalty_flag:
        np.copysign(penalty, eps_z, penalty)
        eps_z += penalty

    return eps_z


def calc_phi_z(g_z, n_avg, sigma, phi_b, u_z_avg=0, p_i=None):
    if p_i is None:
        segments = n_avg
        uniform = True
    else:
        segments = p_i.size
        uniform = segments == round(n_avg)

    g_zs = Propagator(g_z, segments)

    # for terminally attached chains
    if sigma:
        g_zs_ta = g_zs.ta()

        if uniform:
            c_i_ta = sigma / np.sum(g_zs_ta[:, -1])
            g_zs_ta_ngts = g_zs.ngts_u(c_i_ta)
        else:
            c_i_ta = sigma * p_i / np.sum(g_zs_ta, axis=0)
            g_zs_ta_ngts = g_zs.ngts(c_i_ta)

        phi_z_ta = compose(g_zs_ta, g_zs_ta_ngts, g_z)
    else:
        phi_z_ta = 0

    # for free chains
    if phi_b:
        g_zs_free = g_zs.free()

        if uniform:
            r_i = segments
            c_free = phi_b / r_i
            normalizer = np.exp(u_z_avg * r_i)
            c_free = c_free * normalizer
            g_zs_free_ngts = g_zs.ngts_u(c_free)
        else:
            r_i = np.arange(1, segments + 1)
            c_i_free = phi_b * p_i / r_i
            normalizer = np.exp(u_z_avg * r_i)
            c_i_free = c_i_free * normalizer
            g_zs_free_ngts = g_zs.ngts(c_i_free)

        phi_z_free = compose(g_zs_free, g_zs_free_ngts, g_z)
    else:
        phi_z_free = 0

    return phi_z_ta + phi_z_free

def compose(g_zs, g_zs_ngts, g_z):
    prod = g_zs * np.fliplr(g_zs_ngts)
    prod[np.isnan(prod)] = 0
    return np.sum(prod, axis=1) / g_z


class Propagator(object):
    def __init__(self, g_z, segments):
        self.g_z = g_z
        self.shape = int(g_z.size), int(segments)

    def ta(self):
        # terminally attached beginnings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = 0.0
        g_zs[0, 0] = self.g_z[0]
        _calc_g_zs_uniform(self.g_z, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def free(self):
        # free beginnings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = self.g_z
        _calc_g_zs_uniform(self.g_z, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def ngts_u(self, c):
        # free ends of uniform chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c * self.g_z
        _calc_g_zs_uniform(self.g_z, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def ngts(self, c_i):
        # free ends of disperse chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c_i[-1] * self.g_z
        _calc_g_zs(self.g_z, c_i, g_zs, LAMBDA_0, LAMBDA_1)
        return g_zs

    def _new(self):
        return np.empty(self.shape, order='F')
    
    
def _calc_g_zs(g_z, c_i, g_zs, f0, f1):
    coeff = np.array([f1, f0, f1])
    pg_zs = g_zs[:, 0]
    segment_iterator = enumerate(c_i[::-1])
    next(segment_iterator)
    for r, c in segment_iterator:
        g_zs[:, r] = pg_zs = (old_correlate(pg_zs, coeff, 1) + c) * g_z

def _calc_g_zs_uniform(g_z, g_zs, f0, f1):
    coeff = np.array([f1, f0, f1])
    segments = g_zs.shape[1]
    pg_zs = g_zs[:, 0]
    for r in range(1, segments):
        g_zs[:, r] = pg_zs = old_correlate(pg_zs, coeff, 1) * g_z