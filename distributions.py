#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributions - cubic spline fitting + von Mises for torsion angles
"""

import torch
import numpy as np
import pyro
from scipy.stats import vonmises


# Normal Distribution Fit to the Distograms

def calc_moments(distribution, min_distance=2, max_distance=22):
    """
    Calculate mean and standard deviation of a distribution

    Input:
        distribution : 1D torch tensor
        min_distance : int, left border of the distribution (Angstroms)
        max_distance : int, right border of the distribution (Angstroms)

    Output:
        tuple of distribution (mean and standard deviation)
    """
    bins = distribution.shape[0]
    x = torch.linspace(min_distance, max_distance, bins)
    d_mean = torch.sum(x * distribution)
    d_var = torch.sum(distribution * (x - d_mean) ** 2)

    return d_mean, torch.sqrt(d_var)


def normal_distr(x, mu, sigma, s=1):
    """
    Find probability of a value "x" in a normal distribution with mean
    "mu" and standard deviation "sigma"

    Input:
        x     : float, value for which the probability should be calculated
        mu    : mean of the distribution
        sigma : standard deviation of the distribution
        s     : scalar of the distribution
    """

    return s * 1 / (sigma * torch.sqrt(torch.tensor(2 * np.pi))) * torch.exp((-1 / 2) * ((x - mu) / sigma) ** 2)


def fit_normal(distogram):
    """
    This just calculates means and standard deviation of all histograms
    and saves it into a 3D tensor with depth 2
    """
    L = distogram.shape[1]
    params = torch.empty((3, L, L))

    for i in range(L):
        for j in range(L):
            m, s = calc_moments(distogram[:, i, j])
            scalar = torch.max(distogram[:, i, j]) / normal_distr(m, m, s)
            params[0, i, j], params[1, i, j], params[2, i, j] = m, s, scalar

    return params

# cubic spline interpolation
# https://stackoverflow.com/questions/61616810/how-to-do-cubic-spline-interpolation-and-integration-in-pytorch


def torch_searchsort(x, xs):
    """
    Returns index of a bin where xs belongs, where x is a list of bins
    """
    x = torch.cat((torch.tensor([-10.0]), x, torch.tensor([10000.0])))

    for i in range(1, len(x) - 1):
        if x[i - 1] < xs and xs <= x[i]:
            return i - 1


def h_poly_helper(tt):
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=tt[-1].dtype)
    return [sum(A[i, j] * tt[j] for j in range(4)) for i in range(4)]


def h_poly(t):
    tt = [None for _ in range(4)]
    tt[0] = 1
    for i in range(1, 4):
        tt[i] = tt[i - 1] * t
    return h_poly_helper(tt)


def interp(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    I = torch_searchsort(x[1:], xs)
    dx = (x[I + 1] - x[I])
    hh = h_poly((xs - x[I]) / dx)
    return hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx


# NOT USED IN THE OPTIMIZE SCRIPT, BUT MIGHT SERVE AS A GOOD EXTENSION
# If your neural network also predicts the distributions of torsion angles,
# you can fit a differentiable von Mises distributions to them and use it for
# sampling or as another potential. This was also done in the original AlphaFold
# These functions serve exactly that purpose

# Von Mises distribution

# Random number Generator
def randvonmises(angledist, kappa_scalar=1, random_state=1):
    """
    Sample random value from a von Mises distribution fitted to a histogram

    Input:
        angledist    : angle distribution - 1D torch tensor
        kappa_scalar : float, scalar of kappa parameter of von Mises distribution. Controls spread of the distribution
        random_state : int, random seed

    Output:
        randvar : float, one random sample
    """
    np.random.seed(random_state)

    bins = angledist.shape[0]
    xtorsion = torch.linspace(-np.pi, np.pi, bins)

    vmexp = torch.sum(xtorsion * angledist)
    vmvar = torch.sum(angledist * (xtorsion - vmexp) ** 2)

    vmkappa = 1 / vmvar

    randvar = vonmises.rvs(kappa=kappa_scalar * vmkappa, loc=vmexp)
    if randvar < -np.pi:
        randvar = 2 * np.pi + randvar
    elif randvar > np.pi:
        randvar = - 2 * np.pi + randvar
    return randvar


def sample_torsion(phi, psi, kappa_scalar=1, random_state=1):
    """
    Samples one value from each von mises distribution fitted to
    each histogram. Kappa is calculated as 1/var * kappa_scalar, to
    make it more narrow

    Input:
        phi          : torch tensor of shape (bins, Length)
        psi          : torch tensor of shape (bins, Length)
        kappa_scalar : float, scalar of kappa parameter of von Mises distribution. Controls spread of the distribution
        random_state : int, random seed
    """

    phi_sample = torch.tensor(np.round([randvonmises(phi[:, i], kappa_scalar, random_state) for i in range(phi.shape[1])], 4))
    psi_sample = torch.tensor(np.round([randvonmises(psi[:, i], kappa_scalar, random_state) for i in range(psi.shape[1])], 4))
    return phi_sample, psi_sample


# Differentiable von mises
def fit_vm(anglegram, kappa_scalar=8):
    """
    Outputs list of fitted von mises distributions to a specific angleogram
    Each item has a method ".log_prob(x)"

    Input:
        anglegram : torch tensor of shape (bins, Length)
        kappa_scalar : float, scalar of kappa parameter of von Mises distribution. Controls spread of the distribution
    Output:
        distros : list of smooth von Mises distributions
    """
    distros = []
    bins = anglegram.shape[0]
    xtorsion = torch.linspace(-np.pi, np.pi, bins)

    for i in range(anglegram.shape[1]):
        vmexp = torch.sum(xtorsion * anglegram[:, i])
        vmvar = torch.sum(anglegram[:, i] * (xtorsion - vmexp) ** 2)
        vmkappa = kappa_scalar / vmvar

        vm = pyro.distributions.von_mises.VonMises(vmexp, vmkappa)
        distros.append(vm)
    return distros
