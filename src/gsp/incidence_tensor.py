import torch
import numpy as np
import scipy as sp
import networkx as nx
import numpy.linalg as la

from src.multilinear_ops.t2m import t2m

def full_incidence_tensor(A, node_list=None, weight="weight", for_torch=True, device=None):
    """Return the full incidence matrix of a graph with adjacency matrix A.

    Args:
        A (np.array): Adjacency matrix of a graph.

    Returns:
        np.array: Full incidence matrix of the graph.
    """
    if isinstance(A, (nx.Graph, nx.DiGraph)):
        A = nx.to_numpy_array(A, nodelist=node_list, weight=weight)
    elif isinstance(A, sp.sparse.csr_matrix):
        A = A.toarray()
    n = A.shape[0]

    if for_torch:
        i,j = torch.where(A != 0)
        values = torch.hstack([-A[(i,j)], A[(i,j)]])
        indices_minus = torch.vstack([i, i*n + j]) # Different from the original code
        indices_plus = torch.vstack([j, i*n + j])  # The unfolding should be done in the 1st or 3rd mode
                                                # for the incoming and outgoing gradient groupings 
        indices = torch.hstack([indices_minus, indices_plus])
        B = torch.sparse_coo_tensor(indices, values, (n,n**2), device=device)
    else:
        B = np.zeros((n,n,n))
        for i in range(n):
            for j in range(n):
                if A[i,j] != 0:
                    B[i,i,j] = -A[i,j]
                    B[j,i,j] = A[i,j]
    return B


def full_incidence_matrix(A, node_list=None, weight="weight", for_torch=True, device=None, normalized=False):
    if for_torch:
        if isinstance(A, (nx.Graph, nx.DiGraph)):
            A = nx.to_numpy_array(A, nodelist=node_list, weight=weight)
        elif isinstance(A, sp.sparse.csr_matrix):
            A = A.toarray()
        if normalized:
            Deg = np.diag(np.asarray(np.sum(A,axis=1)).ravel())
            Dsq = np.linalg.pinv(np.sqrt(Deg))
            A = Dsq@A@Dsq
        A = torch.tensor(A, device=device)
        n = A.shape[0]
        i,j = torch.where(A != 0)
        values = torch.hstack([-A[(i,j)], A[(i,j)]])
        indices_minus = torch.vstack([i, i*n + j])
        indices_plus = torch.vstack([j, i*n + j])
        indices = torch.hstack([indices_minus, indices_plus])
        B = torch.sparse_coo_tensor(indices, values, (n,n**2), device=device)
        return B
    else:
        B = full_incidence_tensor(A, node_list=None, weight="weight")
        return sp.sparse.csr_array(t2m(B,1))

def torch_incidence_matrix(B, ):
    if isinstance(B, (nx.Graph, nx.DiGraph)):
        B = nx.incidence_matrix(B, oriented=True)
        torch.sparse.