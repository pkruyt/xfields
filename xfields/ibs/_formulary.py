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


def core_std_nplike(nplike, data: ArrayLike) -> float:
    """Compute the std of the core of the beam (20th–80th percentile) using nplike."""
    if len(data) < 2:
        return 0.0
    p20, p80 = nplike.percentile(data, [10, 90])
    core = data[(data >= p20) & (data <= p80)]
    return float(nplike.std(core, ddof=1))


def _beam_intensity(particles: xt.Particles) -> float:
    """Get the beam intensity from the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.sum(particles.weight[particles.state > 0]))


def _bunch_length(particles: xt.Particles) -> float:
    """Get the bunch length from the 20–80% core of the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    data = particles.zeta[particles.state > 0]
    return core_std_nplike(nplike, data)


def _sigma_delta(particles: xt.Particles) -> float:
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    data = particles.delta[particles.state > 0]
    return core_std_nplike(nplike, data)


def _sigma_x(particles: xt.Particles) -> float:
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    data = particles.x[particles.state > 0]
    return core_std_nplike(nplike, data)


def _sigma_y(particles: xt.Particles) -> float:
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    data = particles.y[particles.state > 0]
    return core_std_nplike(nplike, data)


def _gemitt_x(particles: xt.Particles, betx: float, dx: float) -> float:
    """
    Horizontal geometric emittance at a location in the machine,
    for the beta and dispersion functions at this location.
    """
    sigma_x = _sigma_x(particles)
    sig_delta = _sigma_delta(particles)
    return float((sigma_x**2 - (dx * sig_delta) ** 2) / betx)


def _gemitt_y(particles: xt.Particles, bety: float, dy: float) -> float:
    """
    Vertical geometric emittance at a location in the machine,
    for the beta and dispersion functions at this location.
    """
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
    nplike = particles._context.nplike_lib
    px, delta = particles.px[particles.state > 0], particles.delta[particles.state > 0]
    data = px - dpx * delta
    return core_std_nplike(nplike, data)


def _sigma_py(particles: xt.Particles, dpy: float = 0) -> float:
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    py, delta = particles.py[particles.state > 0], particles.delta[particles.state > 0]
    data = py - dpy * delta
    return core_std_nplike(nplike, data)


def _mean_px(particles: xt.Particles, dpx: float = 0) -> float:
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    px: ArrayLike = particles.px[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.mean(px - dpx * delta))


def _mean_py(particles: xt.Particles, dpy: float = 0) -> float:
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
