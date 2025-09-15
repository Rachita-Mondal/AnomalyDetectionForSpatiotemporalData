import numpy as np
import torch
# import jax
# import jax.numpy as jnp

TINY = 1e-32

def prox_l21(x, alpha, axis=0):
    """Applies the proximal operator of the l21 norm with threshold parameter alpha.

    Args:
        x (torch.Tensor): input array
        alpha (float): proximal operator threshold
        axis (int): axis along which to compute the l2 norm

    Returns:
        Tensor: thresholded vectors
    """
    if isinstance(x, np.ndarray):
        norm_l2 = np.linalg.norm(x, axis=axis, keepdims=True)
        scaling_factor = np.where(norm_l2>alpha, 1-alpha/norm_l2, 0)
    elif isinstance(x, torch.Tensor):
        norm_l2 = torch.linalg.vector_norm(x, dim=axis, keepdim=True)
        scaling_factor = torch.where(norm_l2>alpha, 1-alpha/norm_l2, 0)
    return x*scaling_factor

# @jax.jit
# def jax_prox_l21(x, alpha, axis=0):
#     norm_l2 = jnp.linalg.norm(x, axis=0).reshape((1, x.shape[1]))
#     scaling_factor = jnp.zeros(norm_l2.shape)
#     scaling_factor = jnp.where(norm_l2>alpha, 1 - alpha / norm_l2, 0)
#     return x*scaling_factor

# def torch_prox_l21(x, alpha, axis=0):
#     """Applies the proximal operator of the l21 norm with threshold parameter alpha.

#     Args:
#         x (torch.Tensor): input array
#         alpha (float): proximal operator threshold
#         axis (int, tuple of ind): axis along which to compute the l2 norm

#     Returns:
#         Tensor: thresholded vectors
#     """
#     norm_l2 = torch.vector_norm(x, dim=axis, keepdim=True)
#     scaling_factor = torch.where(norm_l2>alpha, 1-alpha/norm_l2, 0)
#     return x*scaling_factor