# TensorAnomalyDetection - a python package for anomaly detection on GraphSignals.
# Copyright (C) 2023 Mert Indibi <indibimu@msu.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import random

import numpy as np
import networkx as nx
from scipy.signal.windows import get_window

from src.multilinear_ops.m2t import m2t
from src.multilinear_ops.t2m import t2m


def generate_sparse_anomaly(x, anomaly_type, amp, ratio=-1, num_of_anomalies=-1, seed=10):
    """Generate synthetic sparse anomaly for the signal x.

    The anomaly is sparse in nature and the entries have iid distribution.

    Args:
        x (np.ndarray): _description_
        anomaly_type (str): determines the type of distribution the anomaly is drawn from.
                            'constant': +M,
                            'bernoulli': +-M,
                            'uniform': U(-M,+M)
                            where M is amplitude
        amp (float): The parameter M in the anomaly_type description. Amplitude of anomaly
        ratio (float): If provided, specifies the number of anomalies to be generated as the
        ratio of the anomalous points to the size of the signal x. Defaults to -1.
        num_of_anomalies (int):  If provided, specifies the number of anomalous points in the
        signal. Defaults to -1.
        seed (int, optional): Random number generator seed. Defaults to 10.        

    Returns:
        anomaly, label (tuple): Synthetic anomaly and the label. label takes on the value 1
        where the anomaly occurs and 0 where anomaly is 0.
    """
    if ratio==-1 and num_of_anomalies==-1:
        raise ValueError("Neither ratio nor the number of anomalies is specified")
    elif (ratio<0 or ratio>1) and ratio!=-1:
        raise ValueError("Ratio of anomalous points cannot be greater than 1 or smaller" +
                         f" than zero. Specified value ratio is {ratio}")
    elif (num_of_anomalies<0 or num_of_anomalies>x.size) and num_of_anomalies!=-1:
        raise ValueError("The number of anomalies cannot be smaller than 0 or"+
                         f" larger than the signal size. x.size={x.size}<{num_of_anomalies}")
    if ratio!=-1 and num_of_anomalies!=-1:
        raise ValueError("You cannot specify both ratio and number of anomalies at the same time.")
    rng = np.random.default_rng(seed)
    anomaly = np.zeros(x.shape)
    if ratio!=-1:
        num_of_anomalies = np.floor(ratio*x.size).astype(int)
    perm = rng.permutation(x.size)
    anomaly_idx = perm[:num_of_anomalies]
    vec_mask = np.zeros(x.size, dtype='bool')
    vec_mask.flat[anomaly_idx]=True
    label = vec_mask.reshape(x.shape)

    if anomaly_type=='constant':
        anomaly.flat[anomaly_idx] = amp
    elif anomaly_type=='bernoulli':
        anomaly.flat[anomaly_idx] = 2*(rng.binomial(n=1,p=0.5,size=(num_of_anomalies,))-0.5)*amp
    elif anomaly_type=='uniform':
        anomaly.flat[anomaly_idx] = rng.uniform(-amp,+amp,size=(num_of_anomalies,))
    return anomaly, label


def generate_local_anomaly(X, G, anomaly_type, amp, local_mode=1, radius=1, num_of_anomalies=1,
                           seed=None, diffusion_coefficient=0.2):
    """Generate locally smooth anomalies for the signal X

    Args:
        X (np.ndarray): Tensor
        G (_type_): _description_
        anomaly_type (_type_): _description_
        amp (_type_): _description_
        local_mode (int, optional): _description_. Defaults to 1.
        radius (int, optional): _description_. Defaults to 1.
        num_of_anomalies (int, optional): _description_. Defaults to -1.
        seed (int, optional): _description_. Defaults to 10.
    """
    A = nx.adjacency_matrix(G)
    # L = nx.laplacianmatrix(G)
    rng = np.random.default_rng(seed)
    anomaly = t2m(np.zeros(X.shape),local_mode)
    anomaly_labels = t2m(np.zeros(X.shape, dtype=bool),local_mode)
    an_time_idices = rng.integers(0, anomaly.shape[1], num_of_anomalies)
    for i in an_time_idices:
        an = np.zeros((anomaly.shape[0],1))
        an_label = np.zeros((anomaly.shape[0],1))
        an_mask= np.zeros((anomaly.shape[0],1), dtype=bool)
        center = rng.integers(0,anomaly.shape[0],1)
        an_label[center]=1

        for j in range(radius):
            an_label += A@an_label
        an_mask = an_label!=0
        if anomaly_type=='constant':
            an[an_mask] = amp
        elif anomaly_type=='bernoulli':
            an[an_mask] = 2*(rng.binomial(n=1,p=0.5)-0.5)*amp
        elif anomaly_type=='uniform':
            an[an_mask]  = rng.uniform(-amp,+amp)
        elif anomaly_type=='diffuse':
            an = rng.uniform(-amp,+amp)*an_label/np.max(np.abs(an_label))
        anomaly_labels[:,i] = np.logical_or(an.ravel(),anomaly_labels[:,i])
        anomaly[:,i] += an.ravel()
    return m2t(anomaly,X.shape,local_mode), m2t(anomaly_labels,X.shape,local_mode)


def generate_temporal_anomaly(x, amplitude, num_of_anomalies, anomaly_duration=4,
                               window_type='boxcar', distribution='constant', temporal_mode=1, 
                               seed=None):
    """Generate temporally contiguous anomalies

    Args:
        x (np.ndarray): Tensor data with time mode specied with temporal_mode
        amplitude (float): The amplitude of the anomalies
        num_of_anomalies (int): number of collective anomalies generated 
        temporal_mode (int, optional): _description_. Defaults to 1.
        window_type (str, optional): The type of window functions generated
        as the anomaly itself. Options are the options of the function
        scipy.signal.windows.get_window. Defaults to 'boxcar'.

    Raises:
        NotImplementedError: _description_

    Returns:
        anomaly (2-tuple): anomaly_signal, anomaly_labels returned as tuple. Labels
        are set to True for the anomalous entries. 
    """
    dims = x.shape
    anomaly = t2m(np.zeros(dims),temporal_mode)
    anomaly_labels = np.zeros(anomaly.shape, dtype=bool)
    M = dims[0]
    N = anomaly_duration
    # generate window function
    w = get_window(window_type, N, fftbins=False)
    # generate random anomaly centers
    # for time t, location l
    rng = np.random.default_rng(seed)
    for i in range(num_of_anomalies):
        t = rng.integers(0,dims[0])
        l = rng.integers(0,dims[1])
        # generate the anomalous entry masks
        if distribution=='constant':
            amp = amplitude
        elif distribution=='bernoulli':
            amp = 2*(rng.binomial(n=1,p=0.5)-0.5)*amplitude
        elif distribution=='uniform':
            amp  = rng.uniform(-amplitude,+amplitude)
        anomaly_labels[np.max((0,t-N//2)):np.min((M,t+N-N//2)),l] = True
        anomaly[np.max((0, t-N//2)):np.min((M, t-N//2+N)),l] += \
            amp*w[-np.min((0, t-N//2)): N- np.max((0, t+N-N//2-M))]    
    return m2t(anomaly, dims, temporal_mode), m2t(anomaly_labels, dims, temporal_mode)


def generate_spatio_temporal_anomaly(dims, G, num_anomalies,amplitude=1, duration=4, radius=1,
                                     window_type='boxcar', distribution='uniform', local_dist='constant',
                                     time_m=2, local_m=1, seed=None, anomaly_spread='isotropic'):
    """Generate randomly scattered spatio-temporally grouped anomalies in a zero tensor.

    Args:
        dims (list of ints): Dimensions of the anomaly tensor
        G (nx.Graph): Graph object representing the spatial domain of the tensor.
        num_anomalies (int): Number of anomaly scattered across the tensor.
        amplitude (float, optional): Amplitude of the scattered anomalies. Defaults to 1.
        duration (int, optional): Time duration of the anomaly. Defaults to 4.
        radius (int, optional): Radius of the anomaly in spatial domain. The spatial anomalies
            are generated centered around the radius-hop neighborhood of the random anomaly center.
            Defaults to 1.
        window_type (str, optional): Window function centered on random anomaly centers. 
            Defaults to 'boxcar' for constant rectangular window.
        distribution (str, optional): Anomaly amplitudes are randomly sampled from given distribution.
            Avaliable options are 'constant', 'bernoulli', 'uniform'. Defaults to 'uniform'.
                'constant': The anomaly amplitudes are equal to `amplitude`
                'bernoulli': The anomaly amplitudes are randomly sampled from {-amplitude, +amplitude}
                'uniform': The anomaly amplitudes are randomly sampled from U(-amplitude, +amplitude)
        local_dist (str, optional): Spatial shape of the anomalies. Avaliable options are
            'constant', 'linear', 'quadratic', 'exponential', 'gaussian'. Defaults to 'constant'.
                The anomaly takes the shape of the window function centered around r-hop neighborhood.
                'constant': The anomaly is constant in space within r-hop radius.
                'linear': The anomaly is linearly decreasing in space w.r.t. the distance from center.
                'quadratic': The anomaly is quadratically decreasing in space w.r.t. the distance from center.
                'exponential': The anomaly is exponentially decreasing in space w.r.t. the distance from center.
                'gaussian': The anomaly is gaussian distributed in space w.r.t. the distance from center.
        anomaly_spread (str, optional): The spread of the anomaly in space. Avaliable options are
            'isotropic', 'anisotropic'. Defaults to 'isotropic'. Isotropic is when the anomalies are
            spread uniformly in all directions. Anisotropic is when the anomalies are spread in random
            walks starting from a center.
        time_m (int, optional): The mode/index of the tensor that correspond to time.
            Defaults to 2.
        local_m (int, optional): The mode/index of the tensor that correspond to space.
            Defaults to 1.
        seed (int, optional): Pseudorandom number generator seed. Defaults to None.

    Returns:
        anomaly_tensor, anomaly_labels: A tuple of the anomaly tensor and the anomaly labels.
    """
    if not isinstance(G,nx.classes.graph.Graph):
        raise TypeError("G is not a networkx graph")
    if len(G)!= dims[local_m-1]:
        raise ValueError("Number of vertices in the graph G does not match indicated the "+
                        f"local_mode dimension. G:{len(G)}!= dims[local_mode]{dims[local_m-1]}")
    if len(dims)<2:
        raise ValueError(f"Specified dimensions are less than two dims={dims}")
    if local_m-1 > len(dims):
        raise IndexError(f"Specified local mode exceeds tensor dimensions. len(dims)={len(dims)}<"+
                        f"local_mode={local_m}")
    if time_m-1 > len(dims):
        raise IndexError(f"Specified time mode exceeds tensor dimensions. len(dims)={len(dims)}<"+
                        f"time_mode={time_m}")
    if time_m == local_m:
        raise IndexError(f"Specified local and time modes are the same.")
    # Swap the axis of time mode to last dimension
    # Swap the axis of local mode to second last dimension
    A = nx.adjacency_matrix(G)
    nodes = list(G.nodes())
    indices = np.arange(len(nodes))
    ind_2_node = dict(zip(indices, nodes))
    node_2_ind = dict(zip(nodes, indices))
    distances = dict(nx.all_pairs_shortest_path_length(G, cutoff=radius))
    random.seed(seed)
    N = len(dims)
    T = dims[time_m-1]
    L = dims[local_m-1]
    D = duration
    dims_t, orig, dest = find_dimension_swaps(dims, local_m, time_m)
    # Generate an anomaly indicator tensor
    # Generate an anomaly tensor
    anomaly = np.zeros(dims_t)
    anomaly_m = np.zeros(dims_t, dtype=bool)
    # Generate random indices of anomaly centers
    rng = np.random.default_rng(seed)
    center_idxs = np.zeros((num_anomalies, N),dtype=int)
    for i in range(num_anomalies):
        center_idxs[i,:] = rng.integers(0,high=dims_t)
    
    # Generate a window function
    if isinstance(duration, int):
        w_t = get_window(window_type, duration, fftbins=False)
    elif isinstance(duration, dict):
        dtn_probabilities = duration['p']
        Ds = duration['d']
        w_ts = [get_window(window_type, d, fftbins=False) for d in Ds]
    else:
        raise ValueError("Duration must be either an integer or a dictionary.")
    w_l = np.zeros(L)
    wl_mask = np.zeros(L, dtype=bool)
    
    # for each anomaly center,
    for i in range(num_anomalies):
        if isinstance(duration, dict):
            D_idx = rng.choice(len(Ds), p=dtn_probabilities)
            D = Ds[D_idx]
            w_t = w_ts[D_idx]
        if distribution=='constant':
            amp = amplitude
        elif distribution=='bernoulli':
            amp = 2*(rng.binomial(n=1,p=0.5)-0.5)*amplitude
        elif distribution=='uniform':
            amp  = rng.uniform(-amplitude,+amplitude)
        t = center_idxs[i,-1]  # Center of the anomaly in time
        el = center_idxs[i,-2] # Center of the anomaly in space
        # Generate a local anomaly
        w_l.fill(0)
        wl_mask.fill(False)
        an_center_node = ind_2_node[el]
            
        if anomaly_spread == 'isotropic':
            # Find the neighbors of the center within the radius
            [wl_mask.__setitem__(node_2_ind[node], True) for node in distances[an_center_node].keys()] 
            

            if local_dist == 'constant': 
                # If the anomaly is constant in space
                w_l[wl_mask] = 1
            elif local_dist == 'linear': 
                # If the anomaly is linearly decreasing in space w.r.t. the distance from center
                [w_l.__setitem__(node_2_ind[node], (1-0.6*distances[an_center_node][node]/(radius))) 
                    for node in distances[an_center_node].keys()];
            elif local_dist == 'quadratic':
                # If the anomaly is quadratically decreasing in space w.r.t. the distance from center
                [w_l.__setitem__(node_2_ind[node], (1-distances[an_center_node][node]/(radius+1))**2) 
                    for node in distances[an_center_node].keys()]
            elif local_dist == 'exponential':
                # If the anomaly is exponentially decreasing in space w.r.t. the distance from center
                [w_l.__setitem__(node_2_ind[node], np.exp(-np.log(10/3)*distances[an_center_node][node]/(radius))) 
                    for node in distances[an_center_node].keys()]
            elif local_dist == 'gaussian':
                # If the anomaly is gaussian distributed in space w.r.t. the distance from center
                [w_l.__setitem__(node_2_ind[node], np.exp(-np.log(2)*distances[an_center_node][node]**2/(radius**2))) 
                    for node in distances[an_center_node].keys()]
        
        elif anomaly_spread == 'anisotropic':
            walk = [an_center_node]
            visited = set([an_center_node])
            distance = [0]  # Distance from start node

            for step_number in range(radius):
                neighbors = [n for n in G.neighbors(walk[-1]) if n not in visited]
                if not neighbors:
                    break
                node = random.choice(neighbors)
                visited.add(node)
                walk.append(node)
                distance.append(distance[-1] + G[walk[-2]][node].get('weight', 1))  # Increase distance by edge weight

            [w_l.__setitem__(node_2_ind[node], local_dist_function(distance[step_number], radius, local_dist)) 
                for step_number, node in enumerate(walk)];
            [wl_mask.__setitem__(node_2_ind[node], True) for node in walk];

        anomaly[tuple(list(center_idxs[i,:-2])+
                      [wl_mask]+
                      [slice(np.max((0, t-D//2)),np.min((T, t-D//2+D)))]
                      )
                      ] += \
                        amp*np.outer(w_l[wl_mask], 
                                     w_t[-np.min((0, t-D//2)): D- np.max((0, t+D-D//2-T))])
        
        # Superpose it to anomaly and anomaly_m tensors
        anomaly_m[tuple(list(center_idxs[i,:-2])+
                        [wl_mask]+
                        [slice(np.max((0, t-D//2)),np.min((T, t-D//2+D)))])] = True 

    # Swap the axis of the anomaly and indicator tensor to have the original shape.
    anomaly = np.moveaxis(anomaly, orig, dest)
    anomaly_m = np.moveaxis(anomaly_m, orig, dest)
    return anomaly, anomaly_m


def find_dimension_swaps(dims, l_m, t_m):
    N = len(dims)
    dim_t = list(dims).copy()
    orig = [i for i in range(N)]
    dest = orig.copy()
    dest.append(dest.pop(l_m-1))
    dim_t.append(dim_t.pop(l_m-1))
    if l_m>t_m:
        dest.append(dest.pop(t_m-1))
        dim_t.append(dim_t.pop(t_m-1))
    else:
        dest.append(dest.pop(t_m-2))
        dim_t.append(dim_t.pop(t_m-2))
    return dim_t, orig, dest 

def random_walk(G, node, L):
    """Generate a random walk of length at most L on the graph G starting from node."""
    walk = [node]

    for i in range(L):
        neighbors = list(G.neighbors(node))
        if not neighbors:
            break
        node = random.choice(neighbors)
        walk.append(node)

    return walk

def local_dist_function(dist, radius, local_dist = 'gaussian'):
    if local_dist == 'constant': 
        # If the anomaly is constant in space
        return 1
    elif local_dist == 'linear': 
        # If the anomaly is linearly decreasing in space w.r.t. the distance from center
        return (1-0.6*dist/(radius))
    elif local_dist == 'quadratic':
        # If the anomaly is quadratically decreasing in space w.r.t. the distance from center
        return (1-dist/(radius+1))**2
    elif local_dist == 'exponential':
        # If the anomaly is exponentially decreasing in space w.r.t. the distance from center
        return np.exp(-np.log(10/3)*dist/(radius))
    elif local_dist == 'gaussian':
        # If the anomaly is gaussian distributed in space w.r.t. the distance from center
        return np.exp(-np.log(2)*dist**2/(radius**2))
    else:
        raise ValueError(f"Local distribution {local_dist} is not supported.")