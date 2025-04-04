# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging

import xobjects as xo
import xtrack as xt
from numpy.typing import ArrayLike

import numpy as np
import scipy.optimize as opt
def gaussian(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

LOGGER = logging.getLogger(__name__)


def phi(beta: ArrayLike, alpha: ArrayLike, dx: ArrayLike, dpx: ArrayLike) -> ArrayLike:
    """
    Computes the ``Phi`` parameter of Eq (15) in
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

    Parameters
    ----------
    beta : ArrayLike
        Beta-function through the machine (chosen plane).
    alpha : ArrayLike
        Alpha-function through the machine (chosen plane).
    dxy : ArrayLike
        Dispersion function through the machine (chosen plane).
    dpxy : ArrayLike
        Dispersion of p[xy] function through the machine (chosen plane).

    Returns
    -------
    phi : ArrayLike
        The ``Phi`` values through the machine.
    """
    return dpx + alpha * dx / beta


# ----- Some helpers on xtrack.Particles objects ----- #


def _beam_intensity(particles: xt.Particles) -> float:
    """Get the beam intensity from the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.sum(particles.weight[particles.state > 0]))


def _bunch_length(particles: xt.Particles) -> float:
    """Get the bunch length from the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.zeta[particles.state > 0]))

def freedman_diaconis_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1 / 3)
    bins = int((np.max(data) - np.min(data)) / bin_width)
    return max(1, bins)  # Ensure at least 1 bin

# def _sigma_delta(particles: xt.Particles) -> float:
#     """
#     Get the standard deviation of the momentum spread
#     from the particles.
#     """
#     _assert_accepted_context(particles._context)
#     nplike = particles._context.nplike_lib
#     data=particles.delta[particles.state > 0]
#     num_bins = freedman_diaconis_bins(data)  
#     bins=num_bins
#     #bins = int(np.sqrt(len(data)))  
#     #bins=10  
#     hist, bin_edges = np.histogram(data, bins=bins, density=True)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
    
#     A_init = max(hist)  
#     mu_init = np.mean(data)  
#     sigma_init = np.std(data)  

#     p0 = [A_init, mu_init, sigma_init]

#     params, covariance = opt.curve_fit(gaussian, bin_centers, hist, p0=p0,maxfev=10000)
#     A_fit, mu_fit, sigma_fit = params
    
#     return float(sigma_fit)

def _sigma_delta(particles: xt.Particles) -> float:
    """
    Get the standard deviation of the momentum spread
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.delta[particles.state > 0]))


def _sigma_x(particles: xt.Particles) -> float:
    """
    Get the horizontal coordinate standard deviation
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.x[particles.state > 0]))


def _sigma_y(particles: xt.Particles) -> float:
    """
    Get the vertical coordinate standard deviation
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.y[particles.state > 0]))


def _gemitt_x(particles: xt.Particles, betx: float, dx: float) -> float:
    """
    Horizontal geometric emittance at a location in the machine,
    for the beta and dispersion functions at this location.
    """
    # Context check is performed in the called functions
    sigma_x = _sigma_x(particles)
    sig_delta = _sigma_delta(particles)
    return float((sigma_x**2 - (dx * sig_delta) ** 2) / betx)


def _gemitt_y(particles: xt.Particles, bety: float, dy: float) -> float:
    """
    Vertical geometric emittance at a location in the machine,
    for the beta and dispersion functions at this location.
    """
    # Context check is performed in the called functions
    sigma_y = _sigma_y(particles)
    sig_delta = _sigma_delta(particles)
    return float((sigma_y**2 - (dy * sig_delta) ** 2) / bety)


def _current_turn(particles: xt.Particles) -> int:
    """
    Get the current tracking turn from one of
    the alive particles.
    """
    _assert_accepted_context(particles._context)
    return int(particles.at_turn[particles.state > 0][0])


def _sigma_px(particles: xt.Particles) -> float:
    """
    Get the horizontal momentum standard deviation
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.px[particles.state > 0]))


def _sigma_py(particles: xt.Particles) -> float:
    """
    Get the vertical momentum standard deviation
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.py[particles.state > 0]))


# ----- Private helper to check the validity of the context ----- #


def _assert_accepted_context(ctx: xo.context.XContext):
    """
    Ensure the context is accepted for IBS computations. We do not
    support PyOpenCL because they have no booleans and lead to some
    wrong results when using boolean array masking, which we do to
    get the alive particles.
    """
    assert not isinstance(ctx, xo.ContextPyopencl), (
        "PyOpenCL context is not supported for IBS. " "Please use either the CPU or CuPy context."
    )
