"""Estimate the degrees of freedom for several models solutions"""

import torch

def naive_generalized_lasso_df(B_indices, D):
    """Generalized sparse lasso degrees of freedom calculation.

    Following the Corollary 2 of the paper,
    - Tibshirani, Ryan J. "The solution path of the generalized lasso". Stanford University, 2011
    Uses matrix rank. There are probably better ways to compute the degrees of freedom.
    Args:
        B_indices (torch.Tensor): A boolean tensor indicating non-zero variables (batch_size, m)
        D (torch.tensor): A tensor of shape (m, p) representing the linear operator/difference matrix.

    Returns:
        float: The estimated degrees of freedom.
    """
    # B_indices = ~B_indices  # Invert the indices to get the complement
    non_zero_b_indices = B_indices.any(dim=1)
    # B_indices = B_indices[non_zero_b_indices]
    df = torch.zeros(B_indices.shape[0], device=B_indices.device, dtype=B_indices.dtype)
    for i in range(B_indices.shape[0]):
        D_tilde = D[~B_indices[i,:]]
        df[i] = D.shape[1] - torch.linalg.matrix_rank(D_tilde)
    return sum(df)