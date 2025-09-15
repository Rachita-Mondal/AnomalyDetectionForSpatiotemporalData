
"""[SNN]-[LOGN+GTV] Decomposition Model

[Sum of nuclear norms]-[Latent Overlapping Group Norm + Graph Total Variation] Decomposition Model
"""
from collections import defaultdict
from time import perf_counter
from math import prod
import warnings

import networkx as nx
import torch
import torch.linalg as la
from torch.nn.functional import softshrink
import matplotlib.pyplot as plt

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.soft_hosvd import soft_moden
from src.proximal_ops.prox_overlapping_grouped_l21 import group_indicator_matrix
from src.gsp.gvr import initialize_graph_variation_regularization
from src.proximal_ops.prox_grouped_l21 import prox_grouped_l21
from src.stats.volume_measures import log_volume_orthogonal_matrix_space

class SNN__LOGN_GTV:
    """[Sum of nuclear norms]-[Latent Overlapping Group Norm + Graph Total Variation] Decomposition Model"""

    def __init__(self, Y, G,
                 lr_modes, graph_modes, gtvr_config,
                 grouping='neighbor', weighing='size_normalized_inv', group_norm_type='l2',
                 soft_constrained=False, **kwargs):
        """Initialize the [SNN]-[LOGS+GTV] decomposition with low-rank modes and (overlapping) grouped norm

        Args:
            Y (torch.Tensor): Tensor to be decomposed.
            G (nx.Graph): Graph structure to define (overlapping) groups
            lr_modes (list of int): List of low-rank modes. 
            graph_modes (list of int): The dimensions over which the graph spans
            gtvr_config (dict): Graph Total Variation Regularization configuration. See class method below.
            grouping (str, optional): Grouping strategy. Defaults to 'neighbor'.
                Options:
                    'edge': creates groupings as the pair of nodes connected by an edge
                    'neighbor': creates groupings with r-hop neighbours of each node.
                        r is specified by the 'r_hop' parameter.
                    'neighbor_plus_center': creates groupings with r-hop neighbours of each node
                        and the node itself as a separate group.
                    'edge_plus_center': creates groupings as the pair of nodes connected by an edge
                        and each node as a separate group.
                    'only_center': Equivalent to ||S||_1
            group_norm_type (str, optional): Type of group norm to use. Defaults to 'l2'.
                Options:
                    'l2': l2 norm of the group
                    'l_inf': l_inf norm of the group (Not Implemented Yet)
            weighing (str, optional): Weighing strategy. Defaults to 'size_normalized_inv'
                Options:
                    'size_normalized': weights the group norms by the square root of the group size
                    'group_size': weights the group norms by the group size
                    'uniform': weights the group norms uniformly
                    'size_normalized_inv': weights the group norms by the inverse of the square root of the group size
                    torch.Tensor: weights the group norms by a custom weighting tensor. 
                        The tensor must have the same length as the number of groups.
            soft_constrained (bool, optional): If True, the observed tensor will be a decomposed into 
                three parts, namely the constraint Y = X + S where X is low rank, S is sparse will be
                turned into Y = X + S + E where E is white gaussian noise. The objective function will have
                + 0.5*lda_f*||E = Y - X - S||^2_F term, representing AWGN noise with 1/lda_f variance.
                Defaults to False.
            kwargs: Additional parameters for the optimization algorithm.
                nodelist (list): The list, whose ordering is used to map the nodes of the graph to
                    graph mode of the tensor.
                edgelist (list): The list, whose ordering is used to map the edges of the graph to
                    graph mode of the tensor.
                r_hop (int): Radius for the grouping strategy 'neighbor'
                device (str): Device to run the model on
                dtype (torch.dtype): Data type to use
                verbose (int): Verbosity level
                metric_tracker (MetricTracker): Metric tracker to log metrics, 
                    find information in src/utils/metric_tracker.py
                report_freq (int): Frequency to report metrics
        """
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.double)
        if isinstance(Y, torch.Tensor):
            self.Y = Y.to(device=self.device, dtype=self.dtype)          # observation matrix
        else:
            self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        self.lr_modes = lr_modes
        self.N = len(self.lr_modes)
        self.graph_modes = graph_modes
        self.graph = G
        self.r_hop = kwargs.get('r_hop', 1)
        self.nodelist = kwargs.get('nodelist', None)
        self.edgelist = kwargs.get('edgelist', None)
        self.grouping = grouping
        self.weighing = weighing
        self.group_norm_type = group_norm_type
        self.soft_constrained = soft_constrained
        self._initialize_groupings()
        self._initialize_variation_regularization(gtvr_config)
        self.X = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # low-rank component
        self.S = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # sparse component
        self._initialize_variables()

        # Bookkeeping
        self.it = 0 
        self.converged = False
        self.times = defaultdict(list)
        self.rhos = defaultdict(list)
        self.obj = []                           # objective function
        self.lagrangian = []                    # lagrangian
        self.r = []
        self.s = []
        self.bic = []
        self.nonzero_params = []
        self.rs = defaultdict(list)
        self.ss = defaultdict(list)
        self.report_freq = kwargs.get('report_freq', 1)
        self.verbose = kwargs.get('verbose', 0)
        self.metric_tracker = kwargs.get('metric_tracker', None)
        
        self.psis = None
        self.lda = None
        self.lda_gtvs = None
        self.lda_f = None
        self.err_tol = None
        self.rho_update = None
        self.rho_update_thr = None


    def __call__(self, psis, lda, lda_gtvs, lda_f=None,
            max_iter=100, rho=0.01, err_tol=1e-6,
            rho_update=1, rho_update_thr=100, **kwargs):
        """Run the [SNN]-[LOGN+GTV] decomposition ADMM algorithm

        Parameters:
        ----------
            psis (list of float): Nuclear norm regularization parameters
            lda (float): Latent Overlapping Group Norm regularization parameter
            lda_gtvs (list of float): Graph Total Variation regularization parameters
            lda_f (float, optional): AWGN noise regularization parameter. Active if soft_constrained is True.
                Defaults to None.
            max_iter (int, optional): Maximum number of iterations for optimization. Defaults to 100.
            rho (float, optional): Augmented lagrangian penalty parameter (AKA step size). Defaults to 0.01.
            err_tol (float, optional): Convergence criteria. Defaults to 1e-6.
            rho_update (float or str, optional): Step size update. Defaults to 1.
                If 'adaptive_spectral' is provided, step-sizes are determined adaptively but has memory cost.
            rho_update_thr (int, optional): Step size update threshold. Defaults to 100.
        
        Returns:
        -------
            torch.Tensor: Low-rank component
            torch.Tensor: Sparse component
        """
        self.psis = psis
        self.lda = lda
        self.lda_gtvs = lda_gtvs
        if self.soft_constrained:
            if lda_f is None:
                raise ValueError("lda_f must be provided if soft_constrained is True")
            self.lda_f = lda_f
        
        self.err_tol = err_tol
        self.rho_update = rho_update
        self.rho_update_thr = rho_update_thr
        self._initialize_rhos(rho)

        while not self.converged and self.it < max_iter:
            tic = perf_counter()
            # First Block Updates:
            ## {X, S_0, S_1,..., S_M, V_1, V_2,..., V_M}
            # if self._rho_update == 'adaptive_spectral':
            #     self._store_first_block()
            self._update_X()
            self._update_Ss()
            self._update_V()
            # if self._rho_update == 'adaptive_spectral':
            #     self._calculate_intermediate_dual()
            #     self._store_second_block()
            # Second Block Updates:
            ## {X_1, ..., X_N, W_1, ..., W_M, (Z_1, ..., Z_M), S}
            self._update_Xs()
            self._update_Ws()
            self._update_Zs()
            self._update_S()
            # Dual Updates
            self._update_duals()
            self.times['iter'].append(perf_counter()-tic)
            self._report_iteration()
            self._check_convergence()
            self._call_tracker()
            if not self.converged:
                self._update_step_size()
                self.it += 1
        return self.X, self.S

    def _update_X(self):
        x_start = perf_counter()
        if not self.soft_constrained:
            self.X = sum([self.rhos[f'X_{i}'][-1]*self.Xs[i] - self.Gamma_Xs[i]
                            for i in range(self.N)]
                        ) + self.rhos['f'][-1]*(self.Y -  self.S - self.Gamma/self.rhos['f'][-1])
            self.X = self.X / (self.rhos['f'][-1] + sum([self.rhos[f'X_{i}'][-1] for i in range(self.N)]))
        else:
            self.X = sum([self.rhos[f'X_{i}'][-1]*self.Xs[i] - self.Gamma_Xs[i]
                            for i in range(self.N)]
                        ) + self.lda_f*(self.Y -  self.S)
            self.X = self.X / (self.lda_f + sum([self.rhos[f'X_{i}'][-1] for i in range(self.N)]))
        self.times['X'].append(perf_counter()-x_start)


    def _update_Ss(self):
        rho_S = [self.rhos[f'S_{j}'][-1] for j in range(self.M+1)]
        rho_Z = self.rhos['Z'][-1]
        rho_W = [self.rhos[f'W_{j}'][-1] for j in range(self.M)]
        for j in range(self.M+1):
            s_start = perf_counter()
            if j == 0:
                Sj_temp = rho_S[j]*(self.S - self.Gamma_Ss[j]/rho_S[j])
                DgZ = tensorize(torch.sum(self.Z, dim=0).coalesce().to_dense(),
                            self.Y.shape, self.graph_modes
                            )
                Sj_temp += rho_Z*(DgZ - self.Gamma_Z/rho_Z)
                self.Ss[j] = Sj_temp/(rho_S[j] + rho_Z)
                self.times[f'S_{j}'].append(perf_counter()-s_start)
            else:
                Sj_rhs = rho_S[j]*matricize(
                                    (self.S - self.Gamma_Ss[j]/rho_S[j]),
                                    self.ops[j-1]['mode']
                                    )+\
                        rho_W[j-1]*(
                            (self.Ws[j-1]- self.Gamma_Ws[j-1]/rho_W[j-1]
                                ).T @ self.ops[j-1]['B.T']
                            ).T 
                self.Ss[j] = tensorize(#torch.sparse.spsolve(Sj_inv, Sj_rhs),
                                torch.linalg.solve(
                                    self.ops[j-1]['Inv'], Sj_rhs),
                                self.Y.shape, self.ops[j-1]['mode']
                                )
                self.times[f'S_{j}'].append(perf_counter()-s_start)
    
    def _update_V(self):
        # V : |Groups| x |Vertices| x Batch (Sparse tensor)
        # Z : |Groups| x |Vertices| x Batch (Sparse tensor)
        # Expander: (|G| x |V| x 1) Hybrid Sparse COO tensor
        v_start = perf_counter()
        rho_v = self.rhos['V'][-1]
        Vtemp = (self.Z - self.expander*(self.Gamma_Vbar/rho_v)).coalesce()
        group_norms = Vtemp.pow(2).sum(dim=1, keepdim=True).coalesce().to_dense().sqrt()
        threshold = self.lda*self.w/rho_v
        scaling_factor = torch.where(group_norms>threshold, 1-threshold/group_norms, 0)
        self.V = Vtemp * scaling_factor
        self.Vbar = self.V.sum(dim=0, keepdim=True).coalesce().to_dense()/self.D_G
        l2_norms = self.V.pow(2).sum(dim=1,keepdim=True).sqrt().to_dense()
        obj = self.lda*torch.sum(self.w*l2_norms)
        self.obj.append(obj.item())
        self.times['V'].append(perf_counter()-v_start)


    def _update_Xs(self):
        rho_x = [self.rhos[f'X_{i}'][-1] for i in range(self.N)]
        for i, m in enumerate(self.lr_modes):
            x_start = perf_counter()
            Xi_temp = (self.X + self.Gamma_Xs[i]/rho_x[i])
            Xi_new, nuc_norm = soft_moden(Xi_temp, self.psis[i]/rho_x[i], m)
            s = torch.norm( Xi_new-self.Xs[i], 'fro')
            self.Xs[i] = Xi_new
            self.obj[-1] += nuc_norm*self.psis[i]
            self.ss[f'X_{i}'].append(rho_x[i]*s)
            self.times[f'X_{i}'].append(perf_counter()-x_start)


    def _update_Ws(self):
        rho_w = [self.rhos[f'W_{j}'][-1] for j in range(self.M)]
        for j in range(self.M):
            w_start = perf_counter()
            Wj_new = self.ops[j]['B.T']@ matricize(self.Ss[j], self.ops[j]['mode'])
            Wj_new += self.Gamma_Ws[j]/rho_w[j]
            # Wj_new, norm = self.ops[j]['prox'](Wj_new, self.lda_gtvs[j]/rho_w[j])
            
            if self.vr_config[j]['variation_type'] == 'GTV' and self.vr_config[j]['p']==2:
                Wj_new, norm = prox_grouped_l21(Wj_new, self.lda_gtvs[j]/rho_w[j], self.ops[j]['E'], return_group_norms=True)
            else:
                Wj_new, norm = softshrink(Wj_new, self.lda_gtvs[j]/rho_w[j]), None
                
            s = torch.norm(
                (Wj_new - self.Ws[j]).T @ self.ops[j]['B.T'],
                'fro')
            if norm is None:
                norm = torch.abs(Wj_new).sum()
            else:
                norm = norm.sum()
            self.obj[-1] += (norm*self.lda_gtvs[j])
            self.Ws[j] = Wj_new
            self.ss[f'W_{j}'].append(rho_w[j]*s)
            self.times[f'W_{j}'].append(perf_counter()-w_start)


    def _update_S(self):
        s_start = perf_counter()
        rho_ss = [self.rhos[f'S_{j}'][-1] for j in range(self.M+1)]
        if not self.soft_constrained:
            rho = self.rhos['f'][-1]
            S_temp = rho*(self.Y - self.X - self.Gamma/rho)
            S_temp += sum([rho_ss[j]*(self.Ss[j] + self.Gamma_Ss[j]/rho_ss[j]) for j in range(self.M+1)])
            S_temp /= (rho + sum(rho_ss))
            s = torch.norm(S_temp - self.S, 'fro')
            self.ss['f'].append(s*rho)
        else:
            S_temp = self.lda_f*(self.Y - self.X)
            S_temp += sum([rho_ss[j]*(self.Ss[j] + self.Gamma_Ss[j]/rho_ss[j]) for j in range(self.M+1)])
            S_temp /= (self.lda_f + sum(rho_ss))
            s = torch.norm(S_temp - self.S, 'fro')
            obj = 0.5*self.lda_f*torch.norm(self.Y - self.X - S_temp, 'fro')**2
            self.obj[-1] += obj
        self.S = S_temp
        for j in range(self.M+1):
            self.ss[f'S_{j}'].append(s*rho_ss[j])
        self.times['S'].append(perf_counter()-s_start)


    def _update_Zs(self):
        z_start = perf_counter()
        rho_z = self.rhos['Z'][-1]
        rho_v = self.rhos['V'][-1]
        Zbar_new = rho_v*(self.Vbar + self.Gamma_Vbar/rho_v)
        Zbar_new += rho_z*(matricize(self.Ss[0] + self.Gamma_Z/rho_z,
                                     self.graph_modes)
                        ) 
        Zbar_new /= (rho_z*self.D_G + rho_v)
        Z_new = ((self.expander*(Zbar_new - self.Vbar)).coalesce() + self.V).coalesce()
        norm = torch.norm(self.Z.values() - Z_new.values(), p='fro')
        self.ss['Z'].append((norm*rho_z))
        self.ss['V'].append((norm*rho_v))
        self.Z = Z_new
        self.Zbar = Zbar_new
        self.times['Z'].append(perf_counter()-z_start)


    def _update_duals(self):
        gamma_start = perf_counter()
        for i in range(self.N):
            r = (self.X - self.Xs[i])
            self.rs[f'X_{i}'].append(torch.norm(r, 'fro'))
            self.Gamma_Xs[i] += self.rhos[f'X_{i}'][-1]*r

        for j in range(self.M+1):
            r = (self.Ss[j] - self.S)
            self.rs[f'S_{j}'].append(torch.norm(r, 'fro'))
            self.Gamma_Ss[j] += self.rhos[f'S_{j}'][-1]*r

        for j in range(self.M):
            r = self.ops[j]['B.T'] @ matricize(
                                self.Ss[j], self.ops[j]['mode']
                                ) - self.Ws[j]
            self.rs[f'W_{j}'].append(torch.norm(r, 'fro'))
            self.Gamma_Ws[j] += self.rhos[f'W_{j}'][-1]*r

        if not self.soft_constrained:
            r = self.S + self.X - self.Y
            self.rs['f'].append(torch.norm(r, 'fro'))
            self.Gamma += self.rhos['f'][-1]*r

        r = self.Ss[0] - tensorize(self.D_G * self.Zbar,
                                    self.Y.shape, self.graph_modes)
        self.rs['Z'].append(torch.norm(r, 'fro'))
        self.Gamma_Z += self.rhos['Z'][-1]*r

        r = self.Vbar - self.Zbar
        self.Gamma_Vbar += self.rhos['V'][-1]*r
        
        r = self.expander * r
        r = torch.norm(r.values())
        self.Gamma_V = (self.expander*self.Gamma_Vbar).coalesce()
        self.rs['V'].append(r*self.rhos['V'][-1])
        self.rs['Z'].append(r*self.rhos['Z'][-1])
        self.times['Gamma'].append(perf_counter()-gamma_start)

        self.r.append(sum([r[-1]**2 for r in self.rs.values()])**(0.5))
        self.s.append(sum([s[-1]**2 for s in self.ss.values()])**(0.5))


    def _initialize_variables(self):
        # Initialize Auxiliary Variables
        self.Xs = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype)
                    for _ in range(self.N)]
        self.Ss = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype)
                    for _ in range(self.M+1)]
        self.Ws = [op['B.T'] @ matricize(torch.zeros_like(self.Y), op['mode']) for op in self.ops]

        self.Zbar = matricize(torch.zeros_like(self.Y),  # 1 x |Vertices| x Batch: Dense tensor
                        self.graph_modes).unsqueeze(0)
        self.Vbar = torch.zeros_like(self.Zbar)          # 1 x |Vertices| x Batch: Dense tensor
                    # Expander: (|G| x |V| x 1) Hybrid Sparse COO tensor
        self.V = (self.expander * self.Zbar).coalesce()  # |Groups| x |Vertices| x Batch: Sparse tensor
        self.Z = (self.expander * self.Zbar).coalesce()  # |Groups| x |Vertices| x Batch: Sparse tensor

        # Initialize Dual Variables
        self.Gamma_Xs = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype)
                    for _ in range(self.N)]
        self.Gamma_Ss = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype)
                    for _ in range(self.M+1)]
        self.Gamma_Ws = [torch.zeros_like(W) for W in self.Ws]

        self.Gamma_Z = torch.zeros_like(self.Y)          
        self.Gamma_Vbar = torch.zeros_like(self.Zbar)         # 1 x |Vertices| x Batch: Dense tensor
        self.Gamma_V = (self.expander * self.Zbar).coalesce() # |Groups| x |Vertices| x Batch: Sparse
        if not self.soft_constrained:
            self.Gamma = torch.zeros_like(self.Y)


    def _initialize_rhos(self, rho):
        """Initialize ADMM Augmented Lagrangian penalty parameters"""
        if isinstance(rho, (int, float)):
            for i in range(self.N):
                self.rhos[f'X_{i}'].append(rho)
            for j in range(self.M+1):
                if j < self.M:
                    self.rhos[f'W_{j}'].append(rho)
                self.rhos[f'S_{j}'].append(rho)
            self.rhos['Z'].append(rho)
            self.rhos['V'].append(rho)
            if not self.soft_constrained:
                self.rhos['f'].append(rho)
        elif isinstance(rho, dict):
            raise NotImplementedError("Custom rho initialization is not implemented yet, please use a scalar value for rho.")
        else:
            raise ValueError("rho must be a scalar or a dictionary.")
        for j in range(self.M):
            self.ops[j]['Inv'] = (self.rhos[f'S_{j}'][-1]*self.ops[j]['I'] +\
                                  self.rhos[f'W_{j}'][-1]*self.ops[j]['BB.T']
                                ).to_dense()
                                # If sparse solver is not available,
                                # use the dense solver


    def _initialize_groupings(self):
        """Initialize groupings based on the specified strategy"""
        G_ind, weights = group_indicator_matrix(self.graph, grouping=self.grouping,
                                                weighing='size_normalized',
                                                r_hop=self.r_hop,
                                                device='cpu',
                                                nodelist=self.nodelist,
                                                edgelist=self.edgelist)
        G_ind = G_ind.to_sparse_coo().to(device=self.device, dtype=torch.float64)
        self.nog = G_ind.shape[0]   # Number of groups
        self.nov = G_ind.shape[1]   # Number of vertices
        self.batch_dim = prod(self.Y.shape)//prod([self.Y.shape[i-1] for i in self.graph_modes])
        if self.weighing == 'size_normalized':
            self.w = weights.to_dense().to(device=self.device, dtype=torch.float64)
        elif self.weighing == 'size_normalized_inv':
            self.w = (1/weights.to_dense()).to(device=self.device, dtype=torch.float64)
        elif self.weighing == 'uniform':
            self.w = torch.ones_like(weights.to_dense()).to(device=self.device, dtype=torch.float64)
        elif isinstance(self.weighing, torch.Tensor):
            self.w = self.weighing.to(device=self.device, dtype=torch.float64)
            if self.w.shape[0] != self.nog:
                raise ValueError("The custom weighting tensor must have"+\
                                 " the same length as the number of groups")
        # w: (|Groups| x 1 x 1) Dense
        self.w = self.w.reshape((self.nog,1,1))
        # Expander: (|Groups| x |Vertices| x 1) Hybrid Sparse COO tensor
        ind = G_ind.indices()
        val = G_ind.to_sparse_coo().values().reshape((-1,1))
        self.expander = torch.sparse_coo_tensor(ind, val, size=(*G_ind.shape,1)).coalesce()
        # D_G: (1 x |Vertices| x 1) Dense tensor
        self.D_G = torch.sum(self.expander, dim=0, keepdim=True).coalesce().to_dense()
        self.group_sizes = torch.sum(self.expander, dim=1).coalesce().to_dense()
    

    def _initialize_variation_regularization(self, vr_config):
        """Initialize graph variation regularization
        
        Parameters:
        ----------
            vr_config (list of dicts): Graph Variation Regularization configurations
                [ { 'graph': ['spatial', 'temporal'],
                    'mode': [List of int]
                    'variation_type': ['GTV', 'GTMV'],
                    'p': [1,2],
                    'normalization': ['out_degree', 'none', ...],
                    'q': 1,},
                    ...]
        """
        self.M = len(vr_config)
        self.vr_config = vr_config
        self.ops = []
        for i, config in enumerate(vr_config):
            op = {'mode': config['mode']}
            if config['graph'] == 'spatial':
                G = self.graph
            elif config['graph'] == 'temporal':
                dim = tuple([self.Y.shape[m-1] for m in config['mode']])
                G = nx.grid_graph(dim, periodic=False)
            else:
                raise ValueError("Only 'spatial' and 'temporal' modes are supported for graph variation regularization.")
            if config['variation_type']=='GTV':
                BT, E = initialize_graph_variation_regularization(G, **config)
                E = E.tocsr()
                op['E'] = torch.sparse_csr_tensor(E.indptr, E.indices, E.data,
                                                device=self.device, dtype=self.dtype)
                p = config.get('p', 2)
                if p == 2:
                    op['prox'] = lambda x, alpha: prox_grouped_l21(x, alpha, op['E'], return_group_norms=True)
                elif p == 1:
                    op['prox'] = lambda x, alpha: (softshrink(x, alpha),None)
            else:
                BT = initialize_graph_variation_regularization(G, **config).tocsr()
                # op['E'] = torch.sparse_csr_tensor(E.indptr, E.indices, E.data,
                #                                 device=self.device, dtype=self.dtype)
                op['prox'] = lambda x, alpha: (softshrink(x, alpha), None)
            BBT = (BT.T @ BT).tocsr()
            op['B.T'] = torch.sparse_csr_tensor(BT.indptr, BT.indices, BT.data,
                                                device=self.device, dtype=self.dtype)
            op['BB.T'] = torch.sparse_csr_tensor(BBT.indptr, BBT.indices, BBT.data,
                                                device=self.device, dtype=self.dtype)
            op['I'] = torch.sparse.spdiags(torch.ones(BBT.shape[0], device='cpu', dtype=self.dtype),
                                        torch.tensor(0, device='cpu'),
                                        BBT.shape, layout=torch.sparse_csr).to(self.device)
            self.ops.append(op)

    

    def _report_iteration(self):
        # bic, k = self.bayesian_information_criterion()
        # self.nonzero_params.append(k)
        # self.bic.append(bic)
        # self.obj[-1] = self.obj[-1].item()
        # self.r[-1] = self.r[-1].item()
        # self.s[-1] = self.s[-1].item()
        self.obj[-1] = (
            self.obj[-1].item()
            if hasattr(self.obj[-1], "item") 
            else self.obj[-1]
        )

        self.r[-1] = (
            self.r[-1].item()
            if hasattr(self.r[-1], "item") 
            else self.r[-1]
        )

        self.s[-1] = (
            self.s[-1].item()
            if hasattr(self.s[-1], "item") 
            else self.s[-1]
        )

        if self.verbose > 0 and self.it % self.report_freq == 0:
            print(f"It-{self.it} \t# |r| = {self.r[-1]:.4e} \t|s| = {self.s[-1]:.4e} \t obj = {self.obj[-1]:.4e} \t {self.times['iter'][-1]:.3f} sec.")
            if self.verbose>1:
                for key in self.rs.keys():
                    print(f"\t# |r_{key}| = {self.rs[key][-1]:.4e} \t|s_{key}| = {self.ss[key][-1]:.4e} \t rho_{key} = {self.rhos[key][-1]:.4e}")


    def _check_convergence(self):
        if self.s[-1] < self.err_tol and self.r[-1] < self.err_tol:
            self.converged = True
            if self.verbose > 1:
                print(f"Converged in {self.it} iterations.")
        return self.converged


    def _call_tracker(self):
        """Tracks metrics of the state variables of the algorithm using a MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)
    
    def _update_step_size(self):
        if not isinstance(self.rho_update, str):
            if self.rho_update <1:
                raise ValueError("Step size growth factor must be larger than 1")
            if self.rho_update != 1.0:
                for key in self.ss.keys():
                    if self.rs[key][-1] > self.ss[key][-1]*self.rho_update_thr:
                        self.rhos[key].append(self.rhos[key][-1]*self.rho_update)
                    elif self.ss[key][-1] > self.rs[key][-1]*self.rho_update_thr:
                        self.rhos[key].append(self.rhos[key][-1]/self.rho_update)
                    else:
                        self.rhos[key].append(self.rhos[key][-1])
        elif self.rho_update == 'adaptive_spectral':
            raise NotImplementedError("Adaptive spectral step size update is not implemented yet.")
        elif self.rho_update == 'domain_parametrization' and self.it %3==0 and self.it<50:
            """Practical use implementation of the domain parametrization step size update.
            
            Taken from the paper:
            'General Optimal Step-size for ADMM-type Algorithms: Domain Parametrization and Optimal Rates', Yifan Ran 2024
            """
            norms = self._calc_variable_norms()
            # first block variables frobenius norm
            first_b_norm = (norms['X']**2+sum([norms[f'S_{j}']**2 for j in range(self.M+1)])+ norms['V']**2)**0.5
            dual_norm = (sum([norms[f'Gamma_X_{i}']**2 for i in range(self.N)]) +\
                        sum([norms[f'Gamma_S_{j}']**2 for j in range(self.M+1)]) +\
                        sum([norms[f'Gamma_W_{j}']**2 for j in range(self.M)]) +\
                        norms['Gamma_Z']**2 + norms['Gamma_V']**2)
            if not self.soft_constrained:
                dual_norm += norms['Gamma']**2
            dual_norm = dual_norm**0.5
            for key in self.ss.keys():
                self.rhos[key].append(dual_norm/first_b_norm)
            
            for j in range(self.M):
                self.ops[j]['Inv'] = (self.rhos[f'S_{j}'][-1]*self.ops[j]['I'] +\
                                    self.rhos[f'W_{j}'][-1]*self.ops[j]['BB.T']
                                    ).to_dense()
                
        
    def _calc_variable_norms(self):
        """Calculates the norms of the variables"""
        norms = {}
        norms['X'] = torch.norm(self.X, 'fro').cpu().item()
        if not self.soft_constrained:
            norms['Gamma'] = torch.norm(self.Gamma, 'fro').cpu().item()
        norms['S'] = torch.norm(self.S, 'fro').cpu().item()
        norms['Z'] = torch.norm(self.Z.values(), 'fro').cpu().item()
        norms['Gamma_Z'] = torch.norm(self.Gamma_Z, 'fro').cpu().item()
        norms['V'] = torch.norm(self.V.values(), 'fro').cpu().item()
        norms['Gamma_V'] = torch.norm(self.Gamma_V.values(), 'fro').cpu().item()
        for i in range(self.N):
            norms[f'X_{i}'] = torch.norm(self.Xs[i], 'fro').cpu().item()
            norms[f'Gamma_X_{i}'] = torch.norm(self.Gamma_Xs[i], 'fro').cpu().item()
        for j in range(self.M+1):
            norms[f'S_{j}'] = torch.norm(self.Ss[j], 'fro').cpu().item()
            norms[f'Gamma_S_{j}'] = torch.norm(self.Gamma_Ss[j], 'fro').cpu().item()
        for j in range(self.M):
            norms[f'W_{j}'] = torch.norm(self.Ws[j], 'fro').cpu().item()
            norms[f'Gamma_W_{j}'] = torch.norm(self.Gamma_Ws[j], 'fro').cpu().item()
        return norms

    def calculate_concentrations(self):
        concentrations = {}
        concentrations['X_F'] = torch.norm(self.X).cpu().numpy().item()
        concentrations['S_F'] = torch.norm(self.S).cpu().numpy().item()
        concentrations['S_1'] = self.S.abs().sum().cpu().numpy().item()
        if concentrations['S_F'] == 0:
            concentrations['S_c'] = 0
        else:
            concentrations['S_c'] = concentrations['S_1']/ (concentrations['S_F']*self.S.numel()**0.5)
        concentration = 0
        for i,m in enumerate(self.lr_modes):
            concentrations[f'X{i+1}_*'] = torch.linalg.svdvals(matricize(self.X, [m])).sum().cpu().numpy().item()
            if concentrations[f'X_F'] == 0:
                concentrations[f'X{i+1}_c'] = 0
            else:
                concentrations[f'X{i+1}_c'] = concentrations[f'X{i+1}_*']/ (concentrations['X_F']*matricize(self.X, [m]).shape[0]**0.5)
            concentration += concentrations[f'X{i+1}_c']
        
        for j in range(self.M):
            
            if self.vr_config[j]['variation_type'] == 'GTV' and self.vr_config[j]['p']==2:
                _, group_norms = prox_grouped_l21(self.Ws[j], 0, self.ops[j]['E'], return_group_norms=True)
                concentrations[f'W_{j+1}_1'] = group_norms.sum().cpu().numpy().item()
                concentrations[f'W_{j+1}_F'] = torch.linalg.norm(group_norms).cpu().numpy().item()
                if concentrations[f'W_{j+1}_F'] == 0:
                    concentrations[f'W_{j+1}_c'] = 0
                else:
                    concentrations[f'W_{j+1}_c'] = concentrations[f'W_{j+1}_1']/ (concentrations[f'W_{j+1}_F']*group_norms.numel()**0.5)
            else:
                concentrations[f'W_{j+1}_1'] = torch.abs(self.Ws[j]).sum()
                concentrations[f'W_{j+1}_F'] = torch.linalg.norm(self.Ws[j]).cpu().numpy().item()
                if concentrations[f'W_{j+1}_F'] == 0:
                    concentrations[f'W_{j+1}_c'] = 0
                else:
                    concentrations[f'W_{j+1}_c'] = concentrations[f'W_{j+1}_1']/ (concentrations[f'W_{j+1}_F']*self.Ws[j].numel()**0.5)
            concentration += concentrations[f'W_{j+1}_c']
        concentrations['Concentration'] = concentration
        return concentrations

    def plot_alg_run(self, figsize=(6,6)):
        """Plots the algorithm log in 2x2 subplots."""
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs = axs.flatten()
        axs[0].semilogy(self.obj)
        axs[0].set_title('Objective function')
        axs[1].semilogy(self.r)
        axs[1].set_title('Primal residual')
        axs[2].semilogy(self.s)
        axs[2].set_title('Dual residual')
        for key in self.rhos.keys():
            axs[3].semilogy(self.rhos[key], label=key)
        axs[3].legend()
        axs[3].set_title(r'Step sizes $(\rho)$')
        for ax in axs:
            ax.grid()
            ax.set_xlabel('Iteration')
        return fig, axs
    

    def bic2(self, cutoff_threshold=1e-6, rank_threshold=0.995):
        """Calculates the Bayesian Information Criterion of the algorithm.
        
        BIC(X,S,E)= #(parameters) * log(p^*) - 2*LogLikelihood(X,S,E)
                #(paramters) = ||S||_0
                                + prod_{n=1}^N r_n                         # Core tensor parameters
                                + sum_{n=1}^N (p_n*r_n - r_n(r_n+1)/2) <-- # Unitary matrices
                -Loglikelihood = Objective function
                                - p^*log(lda_f)
                                - sum_{n=1}^N(p_n*log(lda_nuc_n))
                                - (p^*)*log(lda_gtv_m)
                                - |G|*B*log(lda)
        """
        dim = torch.tensor(self.Y.shape, device=self.device, dtype=self.dtype)
        D = torch.prod(dim)
        t_rank = dim.clone()
        
        obj = self.obj[-1]
        nll = obj
        lda = torch.tensor(self.lda, device=self.device, dtype=self.dtype)
        psis = torch.tensor(self.psis, device=self.device, dtype=self.dtype)
        lda_gtvs = torch.tensor(self.lda_gtvs, device=self.device, dtype=self.dtype)
        if self.soft_constrained:
            lda_f = torch.tensor(self.lda_f, device=self.device, dtype=self.dtype)
            nll -= D*torch.log(lda_f)
        nll -= self.nog*self.batch_dim*torch.log(lda)
        
        
        k = (self.S.abs()>cutoff_threshold).sum().to(dtype=self.dtype)
        for i, m in enumerate(self.lr_modes):
            sv = torch.linalg.svdvals(
                matricize(self.Xs[i], [m])
                )
            total_energy = torch.sum(sv ** 2)
            cumulative_energy = torch.cumsum(sv ** 2, dim=0)
            r = (torch.sum(cumulative_energy < (rank_threshold * total_energy))+1).to(dtype=self.dtype)
            # r = torch.sum(sv > cutoff_threshold*torch.max(sv)).to(dtype=self.dtype)
            n = dim[m-1]
            t_rank[m-1] = r
            k += (r*n - r*(r+1)/2)
            nll -= n*torch.log(psis[i])
        k += torch.prod(t_rank)
        
        for j in range(self.M):
            nll -= D*torch.log(lda_gtvs[j])

        bic = k*torch.log(D)+2*nll
        if self.soft_constrained:
            nll = 0.5*lda_f*torch.norm(self.Y - self.X - self.S, 'fro')**2
            nll -= D*0.5*torch.log(lda_f)
            bic = k*torch.log(D)+2*nll
        return {'NLL': nll.cpu().item(),
                'Objective': obj.cpu().item(),
                'BIC2': bic.cpu().item(),
                'NUM_PARAMETERS': k.cpu().item()}
    

    def bayesian_information_criterion(self, threshold=1e-10):
        """Calculates the Bayesian Information Criterion of the algorithm.
        
        BIC = 2*sum_{i=1}^N ( lda_nuc_i*||X_{(i)}||_* - n_i*log(lda_nuc_i) )
              + 2* (lda1*||S||_1 - D*log(lda1)))
              - k*log(D)
        where k is the total number of non-zero parameters in the estimated X, S variables.
        """
        dim = torch.tensor(self.Y.shape, device=self.device, dtype=self.dtype)
        lda1 = torch.tensor(self.lda, device=self.device, dtype=self.dtype)
        lda_nucs = torch.tensor(self.psis, device=self.device, dtype=self.dtype)
        bic = 0
        k = 0       # Number of non-zero parameters
        for i in range(self.N):
            sv = torch.linalg.svdvals(
                matricize(self.Xs[i], [self.lr_modes[i]])
                )
            #      2 * tau_m * ||X_{(m)}||_*
            bic += 2*lda_nucs[i]*torch.sum(sv)
            #      - n_m * log(tau_m) # I changed the n_m to D = prod(dim)
            bic -= 2*torch.prod(dim)*torch.log(lda_nucs[i]) # dim[self.lr_modes[i]-1] * 
            r = torch.sum(sv > threshold*torch.max(sv))
            n = dim[self.lr_modes[i]-1]
            p = torch.prod(dim)/n
            k += (n+p)*r - r**2

        k += torch.sum(torch.abs(self.S) > threshold)
        bic += 2*lda1*torch.sum(torch.abs(self.S))
        bic -= 2*torch.prod(dim)*torch.log(lda1)
        bic += k*torch.log(torch.prod(dim))
        return bic.cpu().item(), k.cpu().item()
    
    def bayesian_information_criterion_modified(self, threshold=1e-8):
        """Calculates the Bayesian Information Criterion (BIC) of the model.
        
        BIC= 2*NLL(X,S) + k*log(N)
        where NLL(X,S) is the negative log-likelihood of the model,
        k is the number of parameters in the model,
        and N is the number of observations in the data.

        k = (# non-zero groups) * (group size) +
            + sum_{m in modes} (rank(X_m)*(X_m.shape[0] + X_m.shape[1]) - rank(X_m)**2 )
        
        NLL(X,S) = sum_{m in modes} psi_m*||X_m||_* + lambda*||S||_{LOGN}
        """
        dim = torch.tensor(self.Y.shape, device=self.device, dtype=self.dtype)
        D = torch.prod(dim)
        # lda = torch.tensor(self.lda, device=self.device, dtype=self.dtype)
        lda_nucs = torch.tensor(self.psis, device=self.device, dtype=self.dtype)
        bic = 0
        nll = 0
        k = 0       # Number of non-zero parameters
        objective = 0
        rs = []
        nms = []
        for i,m in enumerate(self.lr_modes):
            sv = torch.linalg.svdvals(matricize(self.Xs[i], [m]))
            # log(p(sigma_m | tau_m)) = n_m * log(tau_m) - tau_m * ||X_{(m)}||_*
            obj = lda_nucs[i]*torch.sum(sv)
            objective += obj
            log_p_sigma_m = dim[m-1]*torch.log(lda_nucs[i]) - obj
            nm = dim[m-1]
            
            pm = D//nm
            log_p_U_m = - log_volume_orthogonal_matrix_space(nm,
                                                              int(min(nm, pm).item()))
            log_p_V_m = - log_volume_orthogonal_matrix_space(pm,
                                                              int(min(nm, pm).item()))
            log_p_X_m = log_p_sigma_m + log_p_U_m + log_p_V_m
            nll -= 2*log_p_X_m

            r = torch.sum(sv > threshold*torch.max(sv))
            nms.append(nm)         # Mode dimensions
            rs.append(r)           # Ranks
            k += nm*r - r*(r+1)//2 # Free parameters in the left singular vectors

        # Calculate core tensor dimension
        k += torch.prod(torch.tensor(rs)) * (D//torch.prod(torch.tensor(nms)))
        
        batch_size = self.V.shape[2]
        tau_gs = self.lda*self.w
        log_C_G_g =  - log_volume_orthogonal_matrix_space(self.group_sizes,1)
        log_tau_g = torch.log(tau_gs)
        # group_sizes: (|Groups| x 1 x 1) Dense tensor
        # tau_gs: (|Groups| x 1 x 1) Dense tensor
        # V: |Groups| x |Vertices| x Batch: Sparse tensor
        # l2_norms: (|Groups| x 1 x Batch)
        l2_norms = self.V.pow(2).sum(dim=1,keepdim=True).sqrt().to_dense()
        obj = torch.sum(tau_gs*l2_norms)
        objective += obj
        log_p_V = torch.sum(log_C_G_g + log_tau_g)*batch_size - obj
        nll -= log_p_V
        
        # k += ((l2_norms>threshold)*self.group_sizes).sum()
        k+= torch.sum(self.V.sum(dim=0).coalesce().to_dense()!=0)

        bic = 2*nll + k*torch.log(D)
        return {'BIC': bic.cpu().item(),
                'nonzero_parameters': k.cpu().item(),
                'NLL': nll.cpu().item(),
                'objective': objective.cpu().item()}

    def num_parameters(self, rtol=None):
        """Calculates the number of non-zero parameters in the model."""
        dim = torch.tensor(self.Y.shape, device=self.device, dtype=self.dtype)
        t_rank = dim.clone()

        V_sum = self.V.sum(dim=0, keepdim=True).coalesce().to_dense()
        num_param_S = (V_sum != 0).sum().item()
        # S_ = tensorize(V_sum, self.Y.shape, self.graph_modes)
        # if len(self.vr_config) == 0:
        #     if self.grouping == 'neighbor' and self.r_hop == 0:
        #         # HoRPCA
        #         num_param_S = S_cardinality
        #     else:
        #         # LOGN (Not Properly Implemented Yet)
        #         num_param_S = S_cardinality
        # elif len(self.vr_config) == 1:
        #     if self.vr_config[0]['p'] == 1:
        #         BT = self.ops[0]['B.T']
        #         I = torch.diag(torch.ones(BT.shape[1], device=self.device, dtype=self.dtype))
        #         D = torch.vstack([BT, I])
                
        #         S_ = matricize(S_, self.vr_config[0]['mode'])
        #         nonzero_edge = (self.Ws[0].T!=0) 
        #         nonzero_node = (S_.T!=0)
        #         # print(f"nonzero_node shape: {nonzero_node.shape}, nonzero_edge shape: {nonzero_edge.shape}")
        #         Bindices = torch.hstack([nonzero_edge, nonzero_node])
        #         num_param_S = calculate_df_naive(Bindices, D)
        #     elif self.vr_config[0]['p'] == 2:
        #         warnings.warn('Isotropic GTV degree of freedom is not implemented yet, using isotropic one.', UserWarning)
        #         BT = self.ops[0]['B.T']
        #         I = torch.diag(torch.ones(BT.shape[1], device=self.device, dtype=self.dtype))
        #         D = torch.vstack([BT, I])
                
        #         S_ = matricize(S_, self.vr_config[0]['mode'])

        #         nonzero_edge = (self.Ws[0].T!=0) 
        #         nonzero_node = (S_.T!=0)
        #         # print(f"nonzero_node shape: {nonzero_node.shape}, nonzero_edge shape: {nonzero_edge.shape}")
        #         Bindices = torch.hstack([nonzero_edge, nonzero_node])
        #         num_param_S = calculate_df_naive(Bindices, D)
        #     else:
        #         raise NotImplementedError("(Degree of Freedom) Only p=1 is supported for graph variation regularization.")

        # elif len(self.vr_config) == 2:
        #     if self.vr_config[0]['p'] == 1 and self.vr_config[1]['p'] == 1:
        #         if set(self.vr_config[0]['mode']).intersection(set(self.vr_config[1]['mode'])) == set():
        #             BT1 = self.ops[0]['B.T']
        #             BT2 = self.ops[1]['B.T']
        #             I1 = torch.diag(torch.ones(BT1.shape[1], device=self.device, dtype=self.dtype))
        #             I2 = torch.diag(torch.ones(BT2.shape[1], device=self.device, dtype=self.dtype))
        #             I = torch.diag(torch.ones((I1.shape[0]*I2.shape[0]), device=self.device, dtype=self.dtype))
        #             D = torch.vstack([torch.kron(BT1, I2),
        #                               torch.kron(I1, BT2),
        #                               I])
        #             modes = self.vr_config[0]['mode'] + self.vr_config[1]['mode']

        #             ws_dim = list(self.Y.shape)
        #             ws_dim[0] = BT1.shape[0]
        #             Ws = tensorize(self.Ws[0], ws_dim, self.vr_config[0]['mode'])
        #             nonzero_edges_1 = matricize(Ws, modes).T !=0 # shape (Batch, V_S . V_T)

        #             ws_dim = list(lr_stss.Y.shape)
        #             ws_dim[3] = BT.shape[0]
        #             Ws = tensorize(lr_stss.Ws[1], wt_dim, [4], [1,2,3])
        #             nonzero_time_edge = matricize(Ws, [1,4], [2,3]).T !=0 # shape (Batch, V_S . V_T)
        # else:
        #     raise NotImplementedError("Only 1 or 2 graph variation regularization terms are supported.")
        for i in range(self.N):
            # sv = torch.linalg.svdvals(
            #     matricize(self.Xs[i], [self.lr_modes[i]])
            #     )
            # r = torch.sum(sv > 0)
            r = torch.linalg.matrix_rank(matricize(self.Xs[i], [self.lr_modes[i]]),
                                         rtol=rtol)
            t_rank[self.lr_modes[i]-1] = r
            
        num_param_X = sum(dim*t_rank) + torch.prod(t_rank) - torch.sum(t_rank**2)

        return {'X': num_param_X.item(),
                'X_t_rank': t_rank.cpu().numpy(),
                'S': num_param_S}
    


def calculate_df_naive(B_indices, D):
    """Generalized sparse lasso degrees of freedom calculation.

    Following the Corollary 2 of the paper,
    - Tibshirani, Ryan J. "The solution path of the generalized lasso". Stanford University, 2011
    Args:
        B_indices (torch.Tensor): A boolean tensor indicating non-zero edges and weights with shape (batch_size, m+p)
        D (torch.tensor): A tensor of shape (m+p, p) representing the difference/fusion matrix
    """
    non_zero_b_indices = B_indices.any(dim=1)
    B_indices = B_indices[non_zero_b_indices]
    df = torch.zeros(B_indices.shape[0], device=B_indices.device, dtype=B_indices.dtype)
    for i in range(B_indices.shape[0]):
        D_tilde = D[~B_indices[i,:]]
        df[i] = min(D_tilde.shape) - torch.linalg.matrix_rank(D_tilde)
    return sum(df)
