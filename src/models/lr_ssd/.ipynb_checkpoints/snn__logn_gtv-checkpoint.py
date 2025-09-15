"""[SNN]-[LOGN+GTV] Decomposition Model

[Sum of nuclear norms]-[Latent Overlapping Group Norm + Graph Total Variation] Decomposition Model
"""
from collections import defaultdict
from time import perf_counter
from math import prod

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
                 **kwargs):
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
            kwargs: Additional parameters for the optimization algorithm.
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
        self.grouping = grouping
        self.weighing = weighing
        self.group_norm_type = group_norm_type
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
        self.err_tol = None
        self.rho_update = None
        self.rho_update_thr = None


    def __call__(self, psis, lda, lda_gtvs,
            max_iter=100, rho=0.01, err_tol=1e-6,
            rho_update=1, rho_update_thr=100, **kwargs):
        """Run the [SNN]-[LOGN+GTV] decomposition ADMM algorithm

        Parameters:
        ----------
            psis (list of float): Nuclear norm regularization parameters
            lda (float): Latent Overlapping Group Norm regularization parameter
            lda_gtvs (list of float): Graph Total Variation regularization parameters
            max_iter (int, optional): Maximum number of iterations for optimization. Defaults to 100.
            rho (float, optional): Augmented lagrangian penalty parameter (AKA step size). Defaults to 0.01.
            err_tol (float, optional): Convergence criteria. Defaults to 1e-6.
            rho_update (float, optional): Step size update. Defaults to 1.
            rho_update_thr (int, optional): Step size update threshold. Defaults to 100.
        
        Returns:
        -------
            torch.Tensor: Low-rank component
            torch.Tensor: Sparse component
        """
        self.psis = psis
        self.lda = lda
        self.lda_gtvs = lda_gtvs
        self.err_tol = err_tol
        self.rho_update = rho_update
        self.rho_update_thr = rho_update_thr
        self._initialize_rhos(rho)
        dual_res = []
        while not self.converged and self.it < max_iter:
            tic = perf_counter()
            # First Block Updates:
            ## {X, S_0, S_1,..., S_M, V_1, V_2,..., V_M}
            self._update_X()
            self._update_Ss()
            self._update_V()
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
            dual_res.append(self.s[-1])
            if not self.converged:
                self._update_step_size()
                self.it += 1
        return self.X, self.S, self.r, dual_res, self.obj

    def _update_X(self):
        x_start = perf_counter()
        self.X = sum([self.rhos[f'X_{i}'][-1]*self.Xs[i] - self.Gamma_Xs[i]
                        for i in range(self.N)]
                    ) + self.rhos['f'][-1]*(self.Y -  self.S - self.Gamma/self.rhos['f'][-1])
        self.X = self.X / (self.rhos['f'][-1] + sum([self.rhos[f'X_{i}'][-1] for i in range(self.N)]))
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
        self.obj.append(obj)
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
            Wj_new, norm = self.ops[j]['prox'](Wj_new, self.lda_gtvs[j]/rho_w[j])
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
        rho = self.rhos['f'][-1]
        S_temp = rho*(self.Y - self.X - self.Gamma/rho)
        S_temp += sum([rho_ss[j]*(self.Ss[j] + self.Gamma_Ss[j]/rho_ss[j]) for j in range(self.M+1)])
        S_temp /= (rho + sum(rho_ss))
        s = torch.norm(S_temp - self.S, 'fro')
        self.ss['f'].append(s*rho)
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
                                                device='cpu')
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
    
    def calculate_local_variation_measures(self, summary_stats=False):
        """Local variation measures"""
        lvs = defaultdict(dict)
        for key, X in [('Y', self.Y), ('S', self.S)]: # ('X', self.X)
            for i, vr in enumerate(self.vr_config):
                op = self.ops[i]
                W = op['B.T'] @ matricize(X, op['mode'])
                if vr['variation_type'] == 'GTV':
                    if vr['p'] == 1:
                        lvs[key][f'LV_{i}'] = op['E'] @ torch.abs(W)
                    elif vr['p'] == 2:
                        lvs[key][f'LV_{i}'] = torch.sqrt(op['E'] @ W.pow(2))
                else:
                    lvs[key][f'LV_{i}'] = torch.abs(W)

        if summary_stats:
            stats = {}
            for key in lvs.keys():
                for i, cfg in enumerate(self.vr_config):
                    name = f"{cfg['variation_type']}_{cfg['p']}"
                    stats[f"{key}-{name}-mean"] = lvs[key][f'LV_{i}'].mean().item()
                    stats[f"{key}-{name}-std"]  = lvs[key][f'LV_{i}'].std().item()
                    stats[f"{key}-{name}-min"]  = lvs[key][f'LV_{i}'].min().item()
                    stats[f"{key}-{name}-max"]  = lvs[key][f'LV_{i}'].max().item()
                    stats[f"{key}-{name}-median"]  = lvs[key][f'LV_{i}'].median().item()
                    stats[f"{key}-{name}-q1"]  = lvs[key][f'LV_{i}'].quantile(0.25).item()
                    stats[f"{key}-{name}-q3"]  = lvs[key][f'LV_{i}'].quantile(0.75).item()
                    stats[f"{key}-{name}-IQR"]  = stats[f"{key}-{name}-q3"] - stats[f"{key}-{name}-q1"]
                    stats[f"{key}-{name}-upper_fence"] = stats[f"{key}-{name}-q3"] + 1.5*stats[f"{key}-{name}-IQR"]
            return stats
        return lvs


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
        if self.verbose > 0 and self.it % self.report_freq == 0:
            print(f"It-{self.it} \t# |r| = {self.r[-1]:.4e} \t|s| = {self.s[-1]:.4e} \t BIC:{self.bic[-1]} \t obj = {self.obj[-1]:.4e} \t {self.times['iter'][-1]:.3f} sec.")
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
            for j in range(self.M):
                rho_wj_changed = self.rhos['W_{j}'][-1] != self.rhos['W_{j}'][-2]
                rho_sj_changed = self.rhos['S_{j}'][-1] != self.rhos['S_{j}'][-2]
                if rho_wj_changed or rho_sj_changed:
                    self.ops[j]['Inv'] = (self.rhos[f'S_{j}'][-1]*self.ops[j]['I'] +\
                                          self.rhos[f'W_{j}'][-1]*self.ops[j]['BB.T']
                                        ).to_dense()
                
        
    def _calc_variable_norms(self):
        """Calculates the norms of the variables"""
        norms = {}
        norms['X'] = torch.norm(self.X, 'fro').cpu().item()
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


    def plot_alg_run(self, figsize=(6,6)):
        """Plots the algorithm log in 2x2 subplots."""
        fig, axs = plt.subplots(1, 4, figsize=figsize)
        axs[0].plot(self.obj)
        axs[0].set_title('Objective function')
        axs[1].plot(self.r)
        axs[1].set_title('Primal residual')
        axs[2].plot(self.s)
        axs[2].set_title('Dual residual')
        axs[3].plot(self.rhos)
        axs[3].set_title('Step size')
        for ax in axs:
            ax.grid()
        return fig, axs
    
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



