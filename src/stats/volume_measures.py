import numpy as np
import scipy.special as sps
import torch
import torch.special as tsps


def unit_sphere_area(dim):
    """Return the area of the unit sphere in dimension dim."""
    return 2 * np.pi ** (dim/2)/ sps.gamma(dim/2)

def log_unit_sphere_area(dim):
    """Return the log of the area of the unit sphere in dimension dim."""
    if isinstance(dim, torch.Tensor):
        return dim/2 * torch.log(np.pi, device=dim.device) - tsps.gammaln(dim/2)
    else:
        return dim/2 * np.log(np.pi) - sps.gammaln(dim/2)

def log_volume_orthogonal_matrix_space(n, r=None):
    """Return the log volume of the space of n x r orthonormal matrices. AKA Haar Measure"""
    if r is None:
        r=n.item()
    if isinstance(n, torch.Tensor):
        return r*torch.log(torch.tensor(2, device=n.device))+r*n*torch.log(torch.tensor(np.pi, device=n.device))/2 - tsps.multigammaln(n/2,r)
    else:
        return r*np.log(2)+r*n*np.log(np.pi)/2 - sps.multigammaln(n/2,r)
     