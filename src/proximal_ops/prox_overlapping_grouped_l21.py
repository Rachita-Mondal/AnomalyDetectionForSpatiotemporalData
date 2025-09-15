"""Solves the following Proximal Operator of Overlapping Group l_2/l_1 Norm Regularization

    min.    1/2*||x - v||^2 + lambda_1*||x||_1 + lambda_2*\\sum_{i=1}^{g} w_i*||x_{G_i}||_2
    x \\in R^p

    as defined in the paper:
    Yuan, Lei, Jun Liu, and Jieping Ye. "Efficient methods for overlapping group lasso.",
    Advances in neural information processing systems 24 (2011).

    where G_i is the i-th group, and w_i is the weight for the i-th group.
"""

import torch
import networkx as nx
import scipy as sp
import numpy as np

def prox_overlapping_grouped_l21(V, G_ind, lda1, lda2, weights, hotstart=None,
                                G_ind_T=None, G_ind_coo=None, algo='APGM',
                                err_tol=1e-6, max_iter=1000, verbose=False, 
                                step_size=None, attenuate='constant'):
    """Proximal Operator of Overlapping Group l_2/l_1 Norm Regularization for a batch of vectors
    
    Parameters
    ----------
    V : torch.Tensor
        Input tensor of shape (nob, p)
    G_ind : CSR matrix
        CSR matrix of shape (nog, p) where nog is the number of groups and p is the number of
        variables. Each row of the matrix designates the membership of the variables to the group.
        1 indicates membership. Similar to adjacency matrix designating the edges of a graph. G_ind
        can be constructed using the binary adjacency matrix of a graph or the unoriented Incidence
        matrix of a graph.
    weights : torch.Tensor
        Weights for each group shaped (nog,) Defaults to w_i = sqrt(|G_i|) where |G_i| is the number 
        of variables in the i-th group.
    lda1 : float
        Regularization parameter for l1 norm. Defaults to 0
    lda2 : float
        Regularization parameter for l2 norm
    G_ind_T : CSC matrix
        Transpose of the group indicator matrix. Optional. Defaults to None
    err_tol : float
        Error tolerance for the proximal operator. Defaults to 1e-6
    max_iter : int
        Maximum number of iterations. Defaults to 150
    attenuate : str
        Step size selection method. Defaults to 'BB'
        Options: 'constant', 'sqrt', 'linear', 'BB'
    """
    nob = V.shape[0]            # Number of batches
    # nog = G_ind.shape[0]
    p = G_ind.shape[1]
    # device = V.device
    # dtype = V.dtype
    # one = torch.tensor(1, dtype=dtype, device=device)
    # zero = torch.tensor(0, dtype=dtype, device=device)
    if G_ind_T is None:
        G_ind_T = G_ind.t().to_sparse_csc()
    if G_ind_coo is None:
        G_ind_coo = G_ind.to_sparse_coo()
    ## =--------------------------= l1 norm proximal operator =--------------------------=
    if lda1 >0:
        U = torch.nn.functional.softshrink(V, lda1)
        U_sign = torch.sign(U).reshape(nob,1,p)
        U = torch.abs(U).reshape(nob,p)
    elif lda1 == 0:
        U = torch.abs(V).reshape(nob,p)
        U_sign = torch.sign(U).reshape(nob,1,p)
    else:
        raise ValueError("lda1 must be non-negative")
    ## =---------------------= Batch Pre-thresholding of zero groups =---------------------=
    if lda2 !=0:
        U_t, G_nnz, sup_G_nnz = pre_sift_zero_groups_v1(U, G_ind, weights, lda2, G_ind_T=G_ind_T)
        ## =------------------= Dual Problem =------------------=
        # omega = omega_projector(G_ind_coo, G_nnz) # Omega projector \in R^{nob x nog x p}
        # U_t_ext = U_t.reshape((nob, 1, p))        # U_t extended to R^{nob x 1 x p}
        # For [50, 10000, 10000]  24.7 ms ± 84.2 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)

        problem = ProxDualProblem(U_t, G_ind, G_ind_coo, lda2, weights, G_nnz, err_tol=err_tol,
                                max_iter=max_iter, verbose=verbose, algo='PGM',
                                step_size=step_size, hotstart=hotstart, attenuate=attenuate)
        if algo == 'PGM':
            problem.solve_PGM()
        if algo == 'APGM':
            problem.solve_APGM()
        return torch.squeeze(U_sign*problem.U_tilde), problem.gap, U_t, G_nnz, problem.Y
    else:
        U_t = U
        G_nnz = None # torch.ones((nob, G_ind.shape[0]), dtype=U.dtype, device=U.device)
        sup_G_nnz = None # torch.ones((nob, p), dtype=U.dtype, device=U.device)
        return torch.squeeze(U_sign*U_t), [0], U_t, None, None
    

class ProxDualProblem:
    r"""Solve the dual problem of the proximal operator of the overlapping group l2/l1 norm

    min. {\omega(Y) = -\psi( max( U - Y1, 0), Y)}
    s.t. Y \in \Omega

    or 

    min. 1/2 \|max( U - Y1, 0) - U\| + <max( U - Y1, 0), Y>

    where \psi is the dual function of the proximal operator of the overlapping group l2/l1 norm
    The implementation here is a batch version of the dual problem.

    Parameters
    ----------
    U_t : torch.Tensor
        Pre-thresholded signals of shape (nob, p)
    G_ind : torch.Tensor
        Group indicator matrix of shape (nog, p)
    lda2 : float
        Regularization parameter for group norms
    weights : torch.Tensor
        Weights for each group.
    G_nnz : torch.Tensor
        Non-zero group indicator matrix of shape (nob, nog)
    err_tol : float
        Error tolerance for the proximal operator. Defaults to 1e-6
    max_iter : int
        Maximum number of iterations. Defaults to 150
    verbose : bool
        Print the progress of the algorithm. Defaults to True
    hotstart : torch.Tensor
        Initial guess for the dual variable. Defaults to None
    step_size : float
        Step size for the (accelerated)proximal gradient method. Defaults to None
    algo : str
        Algorithm used to solve the dual problem. Defaults to 'PGM'
        Options: 'PGM' for Proximal Gradient Method, 'APGM' for Accelerated Proximal Gradient Method

    """
    @torch.no_grad
    # @torch.jit.script
    def __init__(self, U_t, G_ind, G_ind_coo, lda2, weights, G_nnz,
                 err_tol=1e-6, max_iter=150, verbose=True,
                 hotstart=None, step_size=None,  algo='APGM', attenuate='constant'):
        self.nob = U_t.shape[0]
        self.p = U_t.shape[1]
        self.nog = G_ind.shape[0]
        self.U_t_ext = U_t.reshape((self.nob, 1, self.p))
        self.wlda2 = weights.to_dense().reshape((1,self.nog, 1))*lda2
        self.G_ind = G_ind
        self.G_ind_coo = G_ind_coo
        self.lda2 = lda2
        self.err_tol = err_tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.G_ind_coo = G_ind_coo
        self.omega = omega_projector(G_ind_coo, G_nnz)  # Omega projector \in R^{nob x nog x p}
        # Used to project the dual variable to the feasible set of the dual problem and
        # calculate the duality gap
        self.algo = algo
        self.gap = []                                   # Duality gap
        self.step_sizes = []
        self.attenuate = attenuate
        self.it = 0
        self.prev_gradient = None
        self.prev_Y = None

        if step_size is None:
            Lc = G_ind.sum(dim=0, keepdim=True).to_dense().max()
            # step_size = 1/(nog**2)
            self._step_size = 1/Lc**2
        else:
            self._step_size = step_size
        if hotstart is not None:
            self.Y = hotstart
        else:
            self.Y = self.omega * 0

        self.U_tilde = self.approximate_primal_solution(self.Y)
        self.gap.append(self.dual_gap(self.Y, self.U_tilde))

    @torch.no_grad
    def approximate_primal_solution(self, Y):
        """Approximates the primal variable from the dual variable"""
        return torch.maximum(self.U_t_ext - torch.sum(Y, dim=1, keepdim=True), torch.tensor(0))

    @torch.no_grad
    def prox_op(self,X):
        r"""Project the dual variable to the feasible set of the dual problem (\Omega)

        Args:
            X (torch.Tensor): Dual variable of shape (nob, nog, p) in sparse COO format

        Returns:
            torch.Tensor: Projected dual variable of shape (nob, nog, p) in sparse COO format
        """
        Y = self.omega * X
        Y_norms = torch.sum(Y.pow(2), dim=2, keepdim=True).sqrt().to_dense()
        ratio = torch.where(Y_norms > self.wlda2, self.wlda2/(Y_norms), 1)
        return ratio * Y

    @torch.no_grad
    def gradient(self, U_tilde):
        """Calculate the gradient of the dual function

        Returns:
            torch.Tensor: Gradient of the dual function of shape (nob, nog, p) in sparse COO format
        """
        return -U_tilde*self.omega

    @torch.no_grad
    def dual_gap(self,Y, U_tilde):
        r"""Calculate the duality gap of the dual problem
        
        gap(Y) = \lambda_2 \sum_{i=1}^{g} w_i ||x_{G_i}||_2 - <Y, x>
        where x = max( U - Y1, 0)
        """
        return torch.sum(
            self.wlda2*(( U_tilde.pow(2)*self.omega).sum(dim=2, keepdim=True).sqrt())
            ) - torch.sum((U_tilde*Y))
    
    @torch.no_grad
    def solve_PGM(self):
        """Solves the problem with Proximal Gradient Method"""

        while self.it < self.max_iter:
            self.it += 1
            gradient = self.gradient(self.U_tilde)
            # self.Y = self.update(self.Y, gradient)
            # self.Y = self.prox_op(self.Y - self.step_size()*gradient)
            # self.U_tilde = self.approximate_primal_solution(self.Y)
            self.Y , self.U_tilde, gap = self.update_PGM(self.Y, gradient)
            self.gap.append(gap)

            if (self.gap[-1] < self.err_tol):
                if self.verbose:
                    print(f"Converged at iteration {self.it} with duality gap {self.gap[-1]:.4E}")
                break
            if self.verbose:
                print(f"Iteration: {self.it}, Gap: {self.gap[-1]:.4E}")
        return self.U_tilde

    @torch.no_grad
    def update_PGM(self, Y, gradient):
        if self.attenuate == 'backtracking_ls':
            pass
        elif self.attenuate == 'BB': # Barzilai-Borwein step size
            if self.it > 1:
                s = Y - self.prev_Y
                y = gradient - self.prev_gradient
                ss = torch.sum(s.pow(2))/torch.sum(s*y)
            else:
                ss = self._step_size
            self.prev_Y = Y.clone()
            self.prev_gradient = gradient.clone()
            bt_it = 0
            while True:
                Y_new = self.prox_op(Y - ss*gradient)
                U_tilde = self.approximate_primal_solution(Y_new)
                gap = self.dual_gap(Y_new, U_tilde)
                
                if gap > (self.gap[-1]+ self.err_tol) and bt_it < 5:
                    ss = ss*0.8
                    bt_it += 1
                    if self.verbose:
                        print(f"Backtracking- Step Size: {ss:.4E}, Gap: {gap:.4E}")
                else:
                    break
            # Y_new = self.prox_op(Y - ss*gradient)
            # U_tilde = self.approximate_primal_solution(Y_new)
            # gap = self.dual_gap(Y_new, U_tilde)
            return Y_new, U_tilde, gap
        else:
            if self.attenuate=='constant':
                ss = self._step_size
            elif self.attenuate=='sqrt':
                ss = self._step_size/(torch.tensor(self.it+1).sqrt())
            elif self.attenuate=='linear':
                ss = self._step_size/(torch.tensor(self.it+1))
            
            Y_new = self.prox_op(Y - ss*gradient)
            U_tilde = self.approximate_primal_solution(Y_new)
            gap = self.dual_gap(Y_new, U_tilde)
            return Y_new, U_tilde, gap
        

    @torch.no_grad
    def solve_APGM(self):
        """Solves the problem with Accelerated Proximal Gradient Method"""
        self.t = torch.tensor(1)         # Momentum term for FISTA
        self.Z = self.Y.clone().detach() # Auxiliary variable for FISTA

        while self.it < self.max_iter:
            
            if self.it == 0:
                self.prev_Y = self.Y.clone()
                gradient = self.gradient(self.U_tilde)
                self.prev_gradient = gradient.clone()
                self.Y = self.prox_op(self.Z - self._step_size*gradient)
                self.U_tilde = self.approximate_primal_solution(self.Y)
                self.gap.append(self.dual_gap(self.Y, self.U_tilde))
                self.ss_prev = self._step_size
            else:
                t_new = (1 + torch.sqrt(1 + 4*self.t**2))/2
                Z_new = self.Y + ((self.t - 1)/t_new)*(self.Y - self.prev_Y)
                gradient = self.gradient(self.approximate_primal_solution(Z_new))

                if self.attenuate == 'BB': # Barzilai-Borwein step size
                    s = self.Y - self.prev_Y
                    y = gradient - self.prev_gradient
                    ss = torch.sum(s.pow(2))/torch.sum(s*y)
                elif self.attenuate == 'constant':
                    ss = self._step_size
                elif self.attenuate == 'sqrt':
                    ss = self._step_size/(torch.tensor(self.it+1).sqrt())
                elif self.attenuate == 'linear':
                    ss = self._step_size/(torch.tensor(self.it+1))
                
                Y_new = self.prox_op(Z_new - ss*gradient)
                self.prev_Y = self.Y.clone()
                self.prev_gradient = gradient.clone()
                self.Y = Y_new
                self.U_tilde = self.approximate_primal_solution(Y_new)
                self.t = t_new
                self.gap.append(self.dual_gap(Y_new, self.U_tilde))

            self.it += 1
            if (self.gap[-1] < self.err_tol):
                if self.verbose:
                    print(f"Converged at iteration {self.it} with duality gap {self.gap[-1]:.4E}")
                break
            if self.verbose:
                print(f"Iteration: {self.it}, Gap: {self.gap[-1]:.4E}")
        return self.U_tilde

    def update_FISTA_with_BB(self, Y, gradient):
        # Check if it's the first iteration to initialize variables
        if self.it == 1:
            self.Z = Y.clone()  # Auxiliary variable for FISTA
            self.t = torch.tensor(1)  # Momentum term for FISTA

        if self.attenuate == 'BB':  # Barzilai-Borwein step size
            if self.it > 1:
                s = Y - self.prev_Y
                y = gradient - self.prev_gradient
                ss = torch.sum(s * s) / torch.sum(s * y)  # BB step size calculation
            else:
                ss = self._step_size  # Default step size for the first iteration
        else:
            # Handle other step size attenuation methods if needed
            ss = self._step_size

        # Update the main variable using the proximal operator and the BB step size
        Y_new = self.prox_op(self.Z - ss * gradient)

        # FISTA update for the auxiliary variable and the momentum term
        t_new = (1 + torch.sqrt(1 + 4 * self.t ** 2)) / 2
        Z_new = Y_new + ((self.t - 1) / t_new) * (Y_new - Y)

        # Update the previous variables for the next iteration
        self.prev_Y = Y.clone()
        self.prev_gradient = gradient.clone()
        self.Z = Z_new
        self.t = t_new

        # Calculate the approximate primal solution and the dual gap
        U_tilde = self.approximate_primal_solution(Y_new)
        gap = self.dual_gap(Y_new, U_tilde)

        return Y_new, U_tilde, gap

def group_indicator_matrix(G, grouping='edge', weighing='size_normalized',
                            device='cuda', dtype=torch.float32, r_hop=1,
                            nodelist=None, edgelist=None):
    """Construct the group indicator matrix from graph G

    Args:
        G (nx.Graph): networkx graph with nodes in V fand edges in E
        grouping (str): method used to construct groups from the graph. Defaults to 'edge'
            Options: 'neighbor', 'edge'
                'neighbor': Groups are the neighbors of the nodes in the graph
                'edge': Groups are the variables connected by the edges in the graph
        weights (str): method used to assign weights to the groups. Defaults to 'degree_normalized'
            Options: 'size_normalized','degree_normalized', 'laplacian_normalized', 'uniform'
                'size_normalized': w_i = sqrt(|G_i|)
                'adjacency_normalized': w = (I + A_n)^-1 @ torch.ones(|V|,1) where A_n is the normalized
                adjacency matrix (for neighbor grouping)
        backend (str): Backend used to construct the group indicator matrix. Defaults to 'torch'
            Options: 'torch', 'scipy', 'jax'

    Returns:
        G_ind (torch.Tensor): Group indicator matrix of shape (nog, |V|)
    """
    if grouping == 'edge':
        B = nx.incidence_matrix(G, oriented=False, nodelist=nodelist, edgelist=edgelist)
        G_ind = torch.sparse_csr_tensor(B.indptr, B.indices, B.data, device=device, dtype=dtype)
    elif grouping == 'neighbor':
        A = nx.adjacency_matrix(G, nodelist=nodelist)
        I = sp.sparse.diags(np.ones(G.number_of_nodes()), format='csr')
        if r_hop == 1:
            A_r = A + I
            G_ind = torch.sparse_csr_tensor(A_r.indptr, A_r.indices, A_r.data, device=device, dtype=dtype) 
        else:
            tmp_A = A.copy()
            A_r = A.copy() + I
            for _ in range(r_hop-1):
                tmp_A = tmp_A @ A
                A_r = A_r + tmp_A
            A_r = A_r>0
            G_ind = torch.sparse_csr_tensor(A_r.indptr, A_r.indices, A_r.data, device=device, dtype=dtype)
    elif grouping == 'edge_plus_center':
        B = nx.incidence_matrix(G, oriented=False, nodelist=nodelist, edgelist=edgelist)
        G_ind = torch.sparse_csr_tensor(B.indptr, B.indices, B.data, device=device, dtype=dtype).to_sparse_coo()
        G_ind = torch.vstack([G_ind, 
                           torch.sparse.spdiags(diagonals= torch.ones(G.number_of_nodes(), device=device, dtype=dtype),
                                      offsets= torch.tensor([0]),
                                      shape=(G.number_of_nodes(), G.number_of_nodes()),
                                      layout=torch.sparse_coo)
                            ]).to_sparse_csr()
    elif grouping == 'neighbor_plus_center':
        A = nx.adjacency_matrix(G, nodelist=nodelist)
        G_ind = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, device=device, dtype=dtype) +\
                torch.sparse.spdiags(diagonals= torch.ones(G.number_of_nodes(), device=device, dtype=dtype),
                                      offsets= torch.tensor([0]),
                                      shape=(G.number_of_nodes(), G.number_of_nodes()),
                                      layout=torch.sparse_csr)
        G_ind = G_ind.to_sparse_coo()
        G_ind = torch.vstack([G_ind,
                            torch.sparse.spdiags(diagonals= torch.ones(G.number_of_nodes(), device=device, dtype=dtype),
                                          offsets= torch.tensor([0]),
                                          shape=(G.number_of_nodes(), G.number_of_nodes()),
                                          layout=torch.sparse_coo)
                             ]).to_sparse_csr()
    elif grouping == 'only_center':
        G_ind = torch.sparse.spdiags(diagonals= torch.ones(G.number_of_nodes(), device=device, dtype=dtype),
                                      offsets= torch.tensor([0]),
                                      shape=(G.number_of_nodes(), G.number_of_nodes()),
                                      layout=torch.sparse_csr)

    
    if weighing == 'size_normalized':
        weights = torch.sum(G_ind, dim=1, keepdims=True).sqrt()
    elif weighing == 'adjacency_normalized':
        Ln = torch.tensor(nx.normalized_laplacian_matrix(G, nodelist=nodelist).toarray(), device=device, dtype=dtype).abs()
        ones = torch.ones(G.number_of_nodes(), device=device, dtype=dtype)
        weights = torch.linalg.solve(Ln, ones)
    
    return G_ind, weights


def pre_sift_zero_groups_v1(U, G_ind, w, lda2, G_ind_T=None):
    """Naive implementation of pre-thresholding of zero groups"""
    # For [50, 10000, 10000] 884 μs ± 715 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    nob = U.shape[0]            # Number of batches
    p = U.shape[1]              # signal dimension
    nog = G_ind.shape[0]        # number of groups
    U_t = U.clone().detach()    # pre-thresholded signals
    dtype = U.dtype; device = U.device
    total_nnz_g = torch.tensor( nog*nob , dtype=dtype, device=device)
    if G_ind_T is None:
        G_ind_T = G_ind.t().to_sparse_csr()
    w = w.to_dense().reshape((1, nog))
    G_nnz = torch.zeros((nob, nog), dtype=dtype, device=device)
    sup_G_nnz = torch.zeros((nob, p), dtype=dtype, device=device)
    one = torch.tensor(1, dtype=dtype, device=device)
    zero = torch.tensor(0, dtype=dtype, device=device)
    while True:
        G_norms = torch.sqrt(torch.mm(G_ind, U_t.pow(2).t())).t()
        torch.where(G_norms > lda2*w, one, zero, out=G_nnz)
        torch.where(torch.mm( G_ind_T, G_nnz.t()).t()> 0, one, zero, out=sup_G_nnz)
        U_t = sup_G_nnz * U_t
        new_total_nnz_g = torch.sum(G_nnz)
        if new_total_nnz_g == total_nnz_g:
            break
        total_nnz_g = new_total_nnz_g
    return U_t, G_nnz, sup_G_nnz


def omega_projector(G_ind_coo, G_nnz):
    """Constructs omega projector from the non-zero group indicator matrix

    Args:
        G_ind_coo (torch.Tensor): Group indicator matrix in sparse COO format of shape (nog, p).
            1 indicates membership of the variable to the group.
        G_nnz (torch.Tensor): Non-zero group indicator matrix of shape (nob, nog). 1 indicates
            the group is non-zero for the batch.

    Returns:
        omega: 
    """
    # For [50, 10000, 10000] 23.8 ms ± 64.3 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    nog = G_ind_coo.shape[0]; nob = G_nnz.shape[0]; p = G_nnz.shape[1]
    device = G_ind_coo.device; dtype = G_ind_coo.dtype

    b_ind = torch.index_select(G_ind_coo, dim=0, index=torch.nonzero(G_nnz[0,:]>0).ravel()).coalesce().indices()
    indices = torch.cat([torch.ones((1,b_ind.shape[1]), dtype=torch.int64, device=device)*0,
                          b_ind],
                         dim=0)

    for i in range(1,nob):
        b_ind = torch.index_select(G_ind_coo, dim=0, index=torch.nonzero(G_nnz[i,:]>0).ravel()).coalesce().indices()
        indices = torch.cat([indices,
            torch.cat([torch.ones((1,b_ind.shape[1]), dtype=torch.int64, device=device)*i, b_ind], dim=0)
        ], dim=1)
    omega = torch.sparse_coo_tensor(indices, torch.ones((indices.shape[1],), dtype=dtype, device=device), (nob, nog, p), device=device)
    return omega

@torch.jit.script
def project_to_omega(omega, X, w2lda2):
    Y = omega * X
    Y_norms = torch.sum(Y.pow(2), dim=2, keepdim=True).sqrt().to_dense()
    ratio = torch.where(Y_norms > w2lda2, w2lda2/(Y_norms), 1)
    return ratio * Y