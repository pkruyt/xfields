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

def freedman_diaconis_bins_nplike(nplike, data):
    """Determine optimal bin count using Freedman–Diaconis rule (nplike compatible)."""
    if len(data) < 2:
        return 1
    q25, q75 = nplike.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1 / 3)
    data_range = nplike.max(data) - nplike.min(data)
    if bin_width == 0 or data_range == 0:
        return 10  # fallback
    bins = int(nplike.maximum(5, nplike.ceil(data_range / bin_width)))
    return bins

def estimate_sigma_from_pdf_peak_nplike(nplike, data):
    """
    Estimate the Gaussian sigma from the maximum of the PDF
    assuming the distribution is approximately Gaussian.
    
    Compatible with CPU and GPU (CuPy) backends via nplike.
    """
    n_bins = freedman_diaconis_bins_nplike(nplike, data)

    # Get bin edges on CPU and transfer if needed
    data_min = float(nplike.min(data))
    data_max = float(nplike.max(data))
    bin_edges_np = np.linspace(data_min, data_max, n_bins + 1)
    bin_edges = nplike.asarray(bin_edges_np)

    # Compute histogram manually (since np.histogram is not nplike-compatible)
    bin_width = bin_edges[1] - bin_edges[0]
    indices = nplike.floor((data - bin_edges[0]) / bin_width).astype(int)
    indices = nplike.clip(indices, 0, n_bins - 1)
    
    counts = nplike.zeros(n_bins, dtype=float)
    for i in range(len(data)):
        counts[indices[i]] += 1

    # Normalize to make it a PDF
    counts = counts / (nplike.sum(counts) * bin_width)

    max_pdf = nplike.max(counts)
    if max_pdf == 0:
        return 0.0

    sigma_est = 1.0 / (np.sqrt(2.0 * np.pi) * max_pdf)
    return float(sigma_est)

def _beam_intensity(particles: xt.Particles) -> float:
    """Get the beam intensity from the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.sum(particles.weight[particles.state > 0]))


def _sigma_delta(particles: xt.Particles) -> float:
    """
    Estimate the core momentum spread (sigma_delta) by using the height of the
    PDF peak, assuming a Gaussian shape, and using Freedman–Diaconis binning.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    delta = particles.delta[particles.state > 0]
    return estimate_sigma_from_pdf_peak_nplike(nplike, delta)


def _bunch_length(particles: xt.Particles) -> float:
    """
    Estimate the bunch length (sigma_zeta) by using the height of the PDF peak
    of zeta, assuming a Gaussian shape, and using Freedman–Diaconis binning.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    zeta = particles.zeta[particles.state > 0]
    return estimate_sigma_from_pdf_peak_nplike(nplike, zeta)



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


def _sigma_px(particles: xt.Particles, dpx: float = 0) -> float:
    """
    Get the horizontal momentum standard deviation from
    the particles. The momentum dispersion can be provided
    to be taken out of the calculation (as we use the stdev
    of px, calling this function at a location with high dpx
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpx : float, optional
        Horizontal momentum dispersion function at the location
        where the sigma_px is computed. Defaults to 0.
    
    Returns
    -------
    sigma_px : float
        The standard deviation of the horizontal momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    px: ArrayLike = particles.px[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.std(px - dpx * delta))


def _sigma_py(particles: xt.Particles, dpy: float = 0) -> float:
    """
    Get the vertical momentum standard deviation from
    the particles. The momentum dispersion can be provided
    to be taken out of the calculation (as we use the stdev
    of py, calling this function at a location with high dpy
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpy : float, optional
        Vertical momentum dispersion function at the location
        where the sigma_py is computed. Defaults to 0.
    
    Returns
    -------
    sigma_py : float
        The standard deviation of the vertical momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    py: ArrayLike = particles.py[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.std(py - dpy * delta))


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
