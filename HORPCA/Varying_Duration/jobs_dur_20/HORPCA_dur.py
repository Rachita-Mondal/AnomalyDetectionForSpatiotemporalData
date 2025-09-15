import optuna
import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
from collections import defaultdict
from matplotlib import pyplot as plt
import optuna
import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
from collections import defaultdict
from matplotlib import pyplot as plt
import argparse
import json
#from dask.distributed import Client, progress, wait
#from IPython.display import display, HTML
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import f1_score

module_path = os.path.abspath(os.path.join('..','..','..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models.lr_ssd.snn__logn_gtv import SNN__LOGN_GTV
from src.models.horpca.horpca_torch import HoRPCA_Singleton
#from dask.distributed import as_completed
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import math
from scipy.optimize import minimize_scalar
from scipy import stats
from scipy.stats import norm
from scipy.stats import multivariate_normal

from sklearn.metrics import precision_recall_curve, auc, roc_curve
from src.synthetic_data.generate_lr_data import generate_low_rank_data
from src.synthetic_data.generate_anomaly import generate_spatio_temporal_anomaly
from src.multilinear_ops.t2m import t2m



def get_data(data_variables):
    seed = data_variables['seed']
    lr_variables = data_variables['lowrank']
    graph_variables = data_variables['graph']
    an_variables = data_variables['anomaly']
    noise = data_variables.get('noise', None)

    X_gt = generate_low_rank_data(**lr_variables, seed=seed)
    std = lr_variables.get('std', None)
    if std is not None:
        X_gt = std*X_gt/ (X_gt.std())

    gtype = graph_variables['type']
    if gtype == 'grid':
        G = nx.grid_2d_graph(graph_variables['n'], graph_variables['m'], 
                             periodic=graph_variables.get('periodic', False))
    elif gtype == 'Gnm':
        G = nx.gnm_random_graph(graph_variables['n'], graph_variables['m'],
                                seed=graph_variables['seed'],
                                directed=graph_variables['directed'])
    elif gtype == 'random_regular':
        G = nx.random_regular_graph(graph_variables['d'], graph_variables['n'],
                                    seed=graph_variables['seed'])
    else:
        raise ValueError('Invalid graph type')
    
    Gt = nx.grid_graph((X_gt.shape[an_variables['time_m']-1], ), periodic=False)
    
    S_gt, labels = generate_spatio_temporal_anomaly(X_gt.shape, G, 
                                                    **an_variables,
                                                    seed=seed)
    
    if noise is not None:
        power_X_db = 10*np.log(np.linalg.norm(X_gt))
        power_N_db = power_X_db - noise['SNR']
        power_N = 10**(power_N_db/10)
        N = np.random.randn(*X_gt.shape)
        N = N/np.linalg.norm(N)*np.sqrt(power_N)
        Y = X_gt + S_gt + N
    else:
        Y = X_gt + S_gt
    return {'Y': Y, 'X_gt': X_gt, 'S_gt': S_gt, 'G':G, 'labels': labels, 'Gt': Gt}


def calculate_metrics(results, data):
    model = results['model']
    model_eps = results['model_eps']
    X_gt = data['X_gt']
    S_gt = data['S_gt']
    labels = data['labels']
    device = model.device
    fpr, tpr, thresholds = roc_curve(labels.ravel(),
                                    torch.abs(model.S).ravel().cpu().numpy())
    precision, recall, thresholds = precision_recall_curve(labels.ravel(),
                                        torch.abs(model.S).ravel().cpu().numpy())
    auc_prc_score = auc(recall, precision)
    auc_roc_score = auc(fpr, tpr)

    metrics = {'BIC': model.bic[-1],
               'AUC-ROC':auc_roc_score,
               'AUC-PRC':auc_prc_score,
               'S_diff': torch.norm(model.S-model_eps.S).cpu().item(),
               'L_diff': torch.norm(model.X-model_eps.X).cpu().item(),
               'L_nuc_1': torch.linalg.matrix_norm(t2m(model.X, 1), 'nuc').cpu().item(),
               'L_nuc_2': torch.linalg.matrix_norm(t2m(model.X, 2), 'nuc').cpu().item(),
               'L_nuc_3': torch.linalg.matrix_norm(t2m(model.X, 3), 'nuc').cpu().item(),
               'S_nuc_1': torch.linalg.matrix_norm(t2m(model.S, 1), 'nuc').cpu().item(),
               'S_nuc_2': torch.linalg.matrix_norm(t2m(model.S, 2), 'nuc').cpu().item(),
               'S_nuc_3': torch.linalg.matrix_norm(t2m(model.S, 3), 'nuc').cpu().item(),
               'L_nuc_1_eps': torch.linalg.matrix_norm(t2m(model_eps.X, 1), 'nuc').cpu().item(),
               'L_nuc_2_eps': torch.linalg.matrix_norm(t2m(model_eps.X, 2), 'nuc').cpu().item(),
               'L_nuc_3_eps': torch.linalg.matrix_norm(t2m(model_eps.X, 3), 'nuc').cpu().item(),
               'S_nuc_1_eps': torch.linalg.matrix_norm(t2m(model_eps.S, 1), 'nuc').cpu().item(),
               'S_nuc_2_eps': torch.linalg.matrix_norm(t2m(model_eps.S, 2), 'nuc').cpu().item(),
               'S_nuc_3_eps': torch.linalg.matrix_norm(t2m(model_eps.S, 3), 'nuc').cpu().item(),
               'L1':  torch.sum(torch.abs(model.X)).cpu().item(),
               'L1_eps':  torch.sum(torch.abs(model_eps.X)).cpu().item(),
               'S1': torch.sum(torch.abs(model.S)).cpu().item(),
               'S1_eps': torch.sum(torch.abs(model_eps.S)).cpu().item(),
               'nonzero_S': (torch.sum(model.S != 0)/torch.prod(torch.tensor(model.S.shape))).cpu().item(),
               'nonzero_L': (torch.sum(model.X != 0)/torch.prod(torch.tensor(model.S.shape))).cpu().item(),
               'ranks_S1': torch.linalg.matrix_rank(t2m(model.S, 1)).cpu().item(),
               'ranks_S2': torch.linalg.matrix_rank(t2m(model.S, 2)).cpu().item(),
               'ranks_S3': torch.linalg.matrix_rank(t2m(model.S, 3)).cpu().item(),
               'ranks_L1': torch.linalg.matrix_rank(t2m(model.X, 1)).cpu().item(),
               'ranks_l2': torch.linalg.matrix_rank(t2m(model.S, 2)).cpu().item(),
               'ranks_l3': torch.linalg.matrix_rank(t2m(model.S, 3)).cpu().item(),
               'S_fro': torch.norm(model.S).cpu().item(),
               'L_fro': torch.norm(model.X).cpu().item(),
               'L_fro_eps': torch.norm(model_eps.X).cpu().item(),
               'S_fro_eps': torch.norm(model_eps.S).cpu().item(),
               'S_err': (torch.norm(torch.tensor(S_gt, device=device)-model.S).cpu().numpy()/np.linalg.norm(S_gt)).item(),
               'L_err': (torch.norm(torch.tensor(X_gt, device=device)-model.X).cpu().numpy()/np.linalg.norm(X_gt)).item()
                }
    metrics['tol']=metrics['S_err']+metrics['L_err'] # tol = S_err + L_err
    metrics['diff'] = metrics['S_diff']+metrics['L_diff'] # diff = S_diff + L_diff
    return metrics


data_variables = {
  'lowrank':{
    'dim': [40, 24, 7, 20],
    'ranks': [8,8,5,5]
  },
  'graph':{
    'type': "grid",
    'n': 8,
    'm': 5,
    'periodic': False,
    'seed': 0,
  },
  'anomaly':{
    'amplitude': 0.25,
    'num_anomalies': 300,
    'duration': 4, # 4, 8, 12, 16, 20, 24 
    'radius': 1,
    'window_type': 'boxcar', #'cosine', 'boxcar'
    'distribution': 'uniform',
    'local_dist': 'constant',
    'time_m': 2,
    'local_m': 1,
    'anomaly_spread': 'isotropic',
  },
   'noise':{
    'type': 'AWGN',
    'SNR': 10,
    },
  'seed':1,
}

data = get_data(data_variables)
## Control Variables
Y = data['Y']
lr_stss_select = {}
lr_stss_select['lr_modes'] = [1,2,3,4]
lr_stss_select['graph_modes'] = [1]
lr_stss_select['grouping'] = 'neighbor'
lr_stss_select['weighing'] = 'uniform'
lr_stss_select['r_hop'] = 0
lr_stss_select['device'] = 'cuda:1'
lr_stss_select['dtype'] = torch.float64
lr_stss_select['verbose'] = 0
lr_stss_select['gtvr_config'] = [{'graph':'spatial',
                       'mode':[1],
                       'p': 1,
                       'variation_type': 'GTMV',
                       'normalization': 'symmetric'},
                       {'graph':'temporal',
                        'mode':[2],
                        'variation_type': 'GTV',
                        'p':1,}]
lr_sts_select = {}
lr_sts_select['lr_modes'] = [1,2,3,4]
lr_sts_select['graph_modes'] = [1]
lr_sts_select['grouping'] = 'neighbor'
lr_sts_select['weighing'] = 'uniform'
lr_sts_select['r_hop'] = 0
lr_sts_select['device'] = 'cuda:1'
lr_sts_select['dtype'] = torch.float64
lr_sts_select['verbose'] = 0
lr_sts_select['gtvr_config'] = [{'graph':'temporal',
                        'mode':[2],
                        'variation_type': 'GTV',
                        'p':1,}]

lr_sss_select = {}
lr_sss_select['lr_modes'] = [1,2,3,4]
lr_sss_select['graph_modes'] = [1]
lr_sss_select['grouping'] = 'neighbor'
lr_sss_select['weighing'] = 'uniform'
lr_sss_select['r_hop'] = 0
lr_sss_select['device'] = 'cuda:1'
lr_sss_select['dtype'] = torch.float64
lr_sss_select['verbose'] = 0
lr_sss_select['gtvr_config'] = [{'graph':'spatial',
                       'mode':[1],
                       'p': 1,
                       'variation_type': 'GTMV',
                       'normalization': 'symmetric'},
                       ]

horpca_select = {}
horpca_select['lr_modes'] = [1,2,3,4]
horpca_select['graph_modes'] = [1]
horpca_select['grouping'] = 'neighbor'
horpca_select['weighing'] = 'uniform'
horpca_select['r_hop'] = 0
horpca_select['device'] = 'cuda:1'
horpca_select['dtype'] = torch.float64
horpca_select['verbose'] = 0
horpca_select['gtvr_config'] = []




def search_hp(data_variables, model_name, n_trials):
    
    data = get_data(data_variables)
    G = data['G']
    S_gt = data['S_gt']
    an_m = data['labels']
    Y = data['Y']
    X = data['X_gt']
    A = nx.adjacency_matrix(G)

    study = optuna.create_study(directions=['maximize', 'maximize'])
    def objective(trial):
        psi = trial.suggest_float('psi', 0, 1)
        lda_1 = 1-psi
        var2 = {}
        var2['psis'] = [psi]*4
        var2['lda'] = 1-psi
        var2['max_iter'] = 100
        var2['rho'] = 4*np.abs(Y).sum()/Y.size
        var2['err_tol'] = 1e-6
        var2['rho_update'] = 1
        var2['rho_update_thr'] = 100
        if model_name == 'lr_stss':
            lda_l = trial.suggest_float('lda_l', 1e-8, 10, log=True)
            lda_t = trial.suggest_float('lda_t', 1e-8, 10, log=True)
            var2['lda_gtvs'] = [lda_l, lda_t]
            model = SNN__LOGN_GTV(Y, G, **lr_stss_select)
        elif model_name == 'lr_sts':
            lda_t = trial.suggest_float('lda_t', 1e-8, 10, log=True)
            lda_l = 0
            var2['lda_gtvs'] = [lda_t]
            model = SNN__LOGN_GTV(Y, G, **lr_sts_select)
        elif model_name == 'lr_sss':
            lda_t = 0
            lda_l = trial.suggest_float('lda_l', 1e-8, 10, log=True)
            var2['lda_gtvs'] = [lda_l]
            model = SNN__LOGN_GTV(Y, G, **lr_sss_select)
        elif model_name == 'horpca':
            lda_t = 0
            lda_l = 0
            var2['lda_gtvs'] = []
            model = SNN__LOGN_GTV(Y, G, **horpca_select)
        
        trial.set_user_attr('psi', psi)
        trial.set_user_attr('lda_1', lda_1)
        trial.set_user_attr('lda_t', lda_t)
        trial.set_user_attr('lda_l', lda_l)
        trial.set_user_attr('number_of_anomalies', noa)
        trial.set_user_attr('anomaly_cardinality', an_m.sum())
        trial.set_user_attr('duration', duration)

        
        L, S = model(**var2)
        L = L.cpu().numpy()
        S = S.cpu().numpy()
        S_loc = t2m(S, m = 1)
        
        r_A = np.eye(S_loc.shape[0])+np.linalg.matrix_power(A.toarray(),radius)
        likelihood = np.zeros(S_loc.shape)

        for s in range(S_loc.shape[0]):
            mask = r_A[np.where(r_A[s,]!=0),:].astype(bool)
            nbd = S_loc[mask[:,0,:].ravel(),:]
            # Append neighbors from additional columns in mask
            for m in range(1, mask.shape[1]):
                nbd = np.vstack((nbd,S_loc[mask[:,m,:].ravel(),:] ))
                
            W = np.zeros(nbd.shape)
            # Iterate through the columns in steps of block_size
            for i1 in range(0, 3360, 140):
                # Slice the matrix to get the block (columns from i to i + block_size)
                if i1==0:
                    block = nbd[:, i1:i1 + 140]
                    for loc in range(nbd.shape[0]):
                        W[loc, i1:i1+140] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1:i1+140]),0,30)
                elif i1==3220:
                    block = nbd[:, i1-140:i1 + 140]
                    for loc in range(nbd.shape[0]):
                        W[loc, i1:i1+140] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-140:i1+140]),0,30)
                else:
                    block = nbd[:, i1-140:i1 + 280]
                    for loc in range(nbd.shape[0]):
                        W[loc, i1:i1+140] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-140:i1+280]),0,30)
                        
            mean = np.sum(W * nbd) / np.sum(W)
            sd = np.sqrt(np.sum(W * (nbd - mean)**2) / np.sum(W))
            likelihood[s,] = np.log(sd) + (0.5*np.power(((S_loc[s,] - mean)/sd),2))
        
        truth = an_m.ravel().reshape(-1,1)
        sig = math.floor(100*(1-(sum(truth)/(len(S.ravel())))))/100
        det = np.zeros(S_loc.shape)
        q = np.quantile(likelihood,sig)
        det[likelihood > q] = 1
        detected_events = det.ravel().reshape(-1,1)
        truth = an_m.ravel().reshape(-1,1)
        roc_auc = roc_auc_score(an_m.ravel(),likelihood.ravel())
        F_score = f1_score(truth, det.reshape(-1,1), average='binary')
        auc = roc_auc_score(an_m.ravel(),np.abs(S).ravel())
        
        trial.set_user_attr('F_score', F_score)
        trial.set_user_attr('roc_auc', roc_auc)
        trial.set_user_attr('auc', auc)
        
        return roc_auc, F_score
    study.optimize(objective, n_trials=n_trials)
    return study

#####################################################################
###########################---Main Code---###########################
#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--path_options', type=json.loads, default={'outvalue':'results_dur_20'},help="optimize options")
parser.add_argument('--iter_options', type=json.loads, default={'rep':0}, help="iter options")
args = parser.parse_args()

if __name__ == '__main__':
    path = '/mnt/home/mondalra/Anomaly_Detection/HORPCA/Varying_Duration/'
    path_out = path+args.path_options['outvalue']+'/'
    rep = args.iter_options['rep']
    models = ['horpca'] #['lr_stss', 'lr_sts', 'lr_sss', 'horpca']
    duration = 20
    n_trials = 500
    n_iteration = 1
    radius = 2
    noa = 450
    res = []
    
    for m, model in enumerate(models):
        print("Running for model: ", model)
        print("Running for model: ", model)
        print("Running for model: ", model)
        seed = rep+1
        data_variables['anomaly']['radius'] = radius
        data_variables['anomaly']['duration'] = duration
        data_variables['anomaly']['num_anomalies'] = noa
        data_variables['seed'] = seed
        study = search_hp(data_variables, model, n_trials)
        best_trial = max(study.best_trials, key=lambda x: x.values[1])
        lda_1_opt = best_trial.user_attrs['lda_1']
        lda_t_opt = best_trial.user_attrs['lda_t']
        lda_l_opt = best_trial.user_attrs['lda_l']
        psi_opt = best_trial.params['psi']
        F_score_opt = best_trial.user_attrs['F_score']
        roc_auc_opt = best_trial.user_attrs['roc_auc']
        res.append({'seed': seed, 'radius': radius, 'duration': duration, 'noa': noa, 'F1_score': F_score_opt, 'roc_auc': roc_auc_opt})
        df = pd.DataFrame(res)
        filename = path_out+'HORPCA_dur_'+str(duration)+'_rep'+str(rep)+'.csv'
        df.to_csv(filename, index=False)
        
        





























