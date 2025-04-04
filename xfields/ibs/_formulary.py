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


def gaussian(x, A, mu, sigma):
    """Gaussian function used for fitting."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def freedman_diaconis_bins(data):
    """Determine optimal bin count using Freedman-Diaconis rule."""
    if len(data) < 2:
        return 1  # Avoid division errors
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1 / 3)
    bins = max(5, int((np.max(data) - np.min(data)) / bin_width))  # At least 5 bins
    return bins

def fit_gaussian(data):
    """Fit Gaussian to histogram with 20–80 percentile filtering; fallback to np.std() if fit fails."""
    
    # Filter to 20th–80th percentile range
    # p20, p80 = np.percentile(data, [20, 80])
    #filtered_data = data[(data >= p20) & (data <= p80)]
    filtered_data=data
    num_bins = freedman_diaconis_bins(filtered_data)
    hist, bin_edges = np.histogram(filtered_data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    A_init = max(hist)
    mu_init = np.mean(filtered_data)
    sigma_init = np.std(filtered_data, ddof=1) or 1e-5
    p0 = [A_init, mu_init, sigma_init]

    try:
        params, _ = opt.curve_fit(gaussian, bin_centers, hist, p0=p0, maxfev=10000)
        return float(params[2])  # Return fitted sigma
    except (RuntimeError, ValueError):
        #LOGGER.warning("Gaussian fit failed; using np.std() instead.")
        return float(sigma_init)


def _beam_intensity(particles: xt.Particles) -> float:
    """Get the beam intensity from the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.sum(particles.weight[particles.state > 0]))


def _bunch_length(particles: xt.Particles) -> float:
    """Get the bunch length using Gaussian fitting; fallback to np.std()."""
    _assert_accepted_context(particles._context)
    return fit_gaussian(particles.zeta[particles.state > 0])

def _sigma_delta(particles: xt.Particles) -> float:
    _assert_accepted_context(particles._context)
    return fit_gaussian(particles.delta[particles.state > 0])


def _sigma_x(particles: xt.Particles) -> float:
    _assert_accepted_context(particles._context)
    return fit_gaussian(particles.x[particles.state > 0])

def _sigma_y(particles: xt.Particles) -> float:
    _assert_accepted_context(particles._context)
    return fit_gaussian(particles.y[particles.state > 0])

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


def _sigma_px(particles: xt.Particles, dpx: float = 0) -> float:
    _assert_accepted_context(particles._context)
    px, delta = particles.px[particles.state > 0], particles.delta[particles.state > 0]
    return fit_gaussian(px - dpx * delta)

def _sigma_py(particles: xt.Particles, dpy: float = 0) -> float:
    _assert_accepted_context(particles._context)
    py, delta = particles.py[particles.state > 0], particles.delta[particles.state > 0]
    return fit_gaussian(py - dpy * delta)


def _mean_px(particles: xt.Particles, dpx: float = 0) -> float:
    """
    Get the arithmetic mean of the horizontal momentum from
    the particles. The momentum dispersion can be provided to
    be taken out of the calculation (as we use the mean of
    px, calling this function at a location with high dpx
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpx : float, optional
        Horizontal momentum dispersion function at the location
        where the mean_px is computed. Defaults to 0.

    Returns
    -------
    mean_px : float
        The arithmetic mean of the horizontal momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    px: ArrayLike = particles.px[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.mean(px - dpx * delta))

def _mean_py(particles: xt.Particles, dpy: float = 0) -> float:
    """
    Get the arithmetic mean of the vertical momentum from
    the particles. The momentum dispersion can be provided to
    be taken out of the calculation (as we use the mean of
    py, calling this function at a location with high dpy
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpy : float, optional
        Vertical momentum dispersion function at the location
        where the mean_py is computed. Defaults to 0.

    Returns
    -------
    mean_py : float
        The arithmetic mean of the horizontal momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    py: ArrayLike = particles.py[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.mean(py - dpy * delta))


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
