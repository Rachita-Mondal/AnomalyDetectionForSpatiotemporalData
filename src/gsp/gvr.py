
"""Graph Variation Regularization (GVR) module."""

import numpy as np
import networkx as nx


def initialize_graph_variation_regularization(G, variation_type='GTV', p=2, normalization='out_degree', q=1, **kwargs):
    """Return graph variation regularization operators for the given graph.

    Parameters:
    ----------
        G (nx.Graph, nx.DiGraph): Graph for which the variation operator is to be computed.
            Graph can be weighted or unweighted.
        variation_type (str): Type of variation operator to be computed. 
            Options: 'GTV': Graph Total Variation (default),
                            GTV_{p,q}(x) = [sum_{u in V}  [sum_{v in N(u)} (w_{u,v}^(1/p) |x_v - x_u|]^p)^(q/p)]^{1/q},
                            where N(u) is the set of neighbors of u, w_{uv} is the weight of the edge (u,v).
                     'GTMV': Graph Total Mean Variation,
                            GTMV(x) = ||Lx||_1,
                            where L is the right normalized weighted graph Laplacian.
                            For directed graphs laplacian is calculated as, L = B@ W @ B^T @ D^{-1},
                            where B is the oriented incidence matrix, W is the diagonal matrix of edge weights,
                            and D is the diagonal matrix of out-degrees.
        p (int, optional): Power of the variation operator. Default is 2.
            Only 1 and 2 are supported.
        normalization (str, optional): Normalization of the variation operator. Default is 'out_degree' for GTV.
            and 'right_normalized' for GTMV.
            Options: 'out_degree', 'in_degree', 'none', 'symmetric'.
                'out_degree': Edge weigths are divided by the out-degree of the source node.
                'in_degree': Edge weigths are divided by the in-degree of the target node.
                'right_normalized': Laplacian is right normalized.
                'symmetric': Laplacian is symmetric normalized
                'left_normalized': Laplacian is left normalized.
                'none': No normalization.
        q (int, optional): Power of the variation operator. Default is 1 and
            other values are not supported.
        **kwargs: Additional keyword arguments.
            node_order (list): List of nodes in the graph to sort variation operator matrices.
            edge_order (list): List of edges in the graph to sort the oriented incidence matrix.
    Returns:
    --------
        Bt or L (csr_matrix): Variation operator matrix.
        E (csr_matrix): Group indicator matrix for GTV variation operator.
            returned when variation_type is 'GTV' and p=2.
            Has the shape (n_nodes, n_edges) and each row has 1s in the columns corresponding to
            the edges belonging to the group. 
    """
    if q != 1:
        raise NotImplementedError("q!=1 is not yet supported for the graph variation regularization.")
    if not G.is_directed() and p == 2:
        G = G.to_directed()
    nodes = kwargs.get('node_order', list(G.nodes))
    edges = kwargs.get('edge_order', list(G.edges))
    if nx.is_weighted(G):
        w = np.array([G[u][v]['weight'] for u,v in edges])
    else:
        w = np.ones(len(edges))
    B = nx.incidence_matrix(G, oriented=True, nodelist=nodes, edgelist=edges)
    if variation_type == 'GTV':
        if normalization == 'out_degree':
            w = w/np.array([np.array([G.out_degree(u) for u,v in edges])])
        elif normalization == 'in_degree':
            w = w/np.array([np.array([G.in_degree(v) for u,v in edges])])
        elif normalization == 'none':
            pass
        Bt = B.multiply(np.power(w.reshape((1,-1)), 1/p)
                        ).tocsc().T
        if p == 2:
            group_indicator = ((B <0)*1.0).tocsr()
            return Bt, group_indicator
        elif p == 1:
            group_indicator = ((B <0)*1.0).tocsr()
            return Bt, group_indicator
        else:
            raise NotImplementedError("p!=1,2 is not yet supported for the graph variation regularization.")
    elif variation_type == 'GTMV':
        L = (B.multiply(w.reshape((1,-1))) @ B.T)
        d = L.diagonal()
        d[d==0] = 1
        if normalization == 'right_normalized':
            Lnr = (L / d.reshape((-1,1)))
        elif normalization == 'left_normalized':
            Lnr = (L / d.reshape((1,-1)))
        elif normalization == 'symmetric':
            Lnr = (L / np.sqrt(d).reshape((-1,1))
                    )/np.sqrt(d).reshape((1,-1))
        elif normalization == 'none':
            Lnr = L
        else:
            raise ValueError("Normalization type not recognized.")
        Lnr.tocsr()
        return Lnr
