import os, sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import optuna
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from scipy.stats import norm
import torch
import argparse
import json

BASE_DIR = Path.cwd().resolve().parent.parent
SCRIPT_DIR = Path.cwd().resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
SMD_DIR = DATA_DIR / 'server_machine_dataset'
SMD_TRAIN_DIR = SMD_DIR / 'train'
SMD_TEST_DIR = SMD_DIR / 'test'
SMD_TEST_LABEL_DIR = SMD_DIR / 'test_label'
SMD_TEST_INTERPRETATION_LABEL_DIR = SMD_DIR / 'interpretation_label'


sys.path.append(BASE_DIR.as_posix())

from smd_anomaly_detection.data.server_machine_dataset import SMDMachineChannel

from src.models.lr_ssd.snn__logn_gtv import SNN__LOGN_GTV
#from src.stats.multi_linear_normal import MultiLinearNormal
from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize


parser = argparse.ArgumentParser()
parser.add_argument('--machine_id',
                    type=int, default=1,
                    help='machine id')
parser.add_argument('--channel_id',
                    type=int, default=1,
                    help='channel id')

args = parser.parse_args()

day_folding = (24,60)
machine_id = args.machine_id
channel_id = args.channel_id
mchannel = SMDMachineChannel(machine_id,channel_id)
Y = mchannel.Y
labels = mchannel.labels
rank_threshold = 0.99
fig, axe = plt.subplots(1,4, figsize=(20, 5))
for i in range(4):
    svals = np.linalg.svd(matricize(Y, [i+1]), compute_uv=False)
    axe[i].plot(svals, marker='o')
    total_energy = np.sqrt(np.sum(svals**2))
    cumulative_energy = np.sqrt(np.cumsum(svals**2))
    r = (np.sum(cumulative_energy < (rank_threshold * total_energy))+1)
    print(f"Mode-{i} approx rank: {r}/{svals.shape[0]}")
    print(f"Estimated noise variance from mode-{i}: {svals[-1]**2/(matricize(Y, [i+1]).shape[1])}")

G = nx.from_scipy_sparse_array(kneighbors_graph(mchannel.test_data, 2))
nx.draw_kamada_kawai(G, labels={i: f'{i+1}' for i in range(38)})
Gt = nx.grid_graph(dim=(day_folding[1],))

def likelihood_scoring(model, G, sig=0.99):
    S_loc = matricize(model.Y - model.X, [1]).cpu().numpy()
    radius = 2
    r_A = np.eye(S_loc.shape[0])+np.linalg.matrix_power(nx.adjacency_matrix(G).toarray(),radius)
    likelihood = np.zeros(S_loc.shape)
    for s in range(S_loc.shape[0]):
        mask = r_A[np.where(r_A[s,]!=0),:].astype(bool)
        nbd = S_loc[mask[:,0,:].ravel(),:]
        # Append neighbors from additional columns in mask
        for m in range(1, mask.shape[1]):
            nbd = np.vstack((nbd,S_loc[mask[:,m,:].ravel(),:] ))
                
        W = np.zeros(nbd.shape)
        # Iterate through the columns in steps of block_size
        for i1 in range(0, S_loc.shape[1], 1440):
            # Slice the matrix to get the block (columns from i to i + block_size)
            if i1==0:
                block = nbd[:, i1:i1 + 1440]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+1440] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1:i1+1440]),0,30)
            elif i1 == S_loc.shape[1]-1440: #23040:
                block = nbd[:, i1-1440:i1 + 1440]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+1440] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-1440:i1+1440]),0,30)
            else:
                block = nbd[:, i1-1440:i1 + 2880]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+1440] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-1440:i1+2880]),0,30)
                        
        mean = np.sum(W * nbd) / np.sum(W)
        sd = np.sqrt(np.sum(W * (nbd - mean)**2) / np.sum(W))
        if sd == 0:
            sd = sd+(10e-20)
        likelihood[s,] = np.log(sd) + (0.5*np.power(((S_loc[s,] - mean)/sd),2))

    det = np.zeros(S_loc.shape)
    q = np.quantile(likelihood,sig)
    det[likelihood > q] = 1

    scoring_scores = mchannel.anomaly_scoring_score(tensorize(likelihood, model.Y.shape, [1]))
    detection_scores = mchannel.anomaly_detection_score(tensorize(det, model.Y.shape, [1]))

    return scoring_scores, detection_scores


max_iter = 300
lda_f = 300
rho = 0.1
gtvr_config = [ { 'graph': 'spatial',
                    'mode': [1],
                    'variation_type': 'GTMV',
                    'p': 1,
                    'normalization': 'none'
                    }]
study = optuna.create_study(directions=['maximize', 'maximize'])
def lrsss_objective(trial):
    psi = trial.suggest_float('psi', 0, 1)
    psis = [psi]*4
    lda = 1-psi
    lda_s = trial.suggest_float('lda_s', 1e-8, 10, log=True)
    # lda = trial.suggest_float("lda", 0.001, 20, log=True)
    # psis = [trial.suggest_float(f"psi_{i}", 0.01, 20, log=True) for i in range(1, 5)]
    # lda_s = trial.suggest_float("lda_s", 1e-6, 0.1, log=True)
    lr_sss= SNN__LOGN_GTV(Y, G, lr_modes=[1,2,3,4], graph_modes=[1], gtvr_config=gtvr_config,
                      grouping='neighbor', r_hop=0,
                      soft_constrained=True, device='cuda:1', verbose=False)

    X, S = lr_sss(psis=psis, lda=lda, lda_gtvs=[lda_s], lda_f=lda_f,
                  rho_update='domain_parametrization', max_iter=max_iter, rho=rho, tol=1e-4)
    
    V_sum = lr_sss.V.sum(dim=0, keepdim=False).coalesce().to_dense()
    S_ = tensorize(V_sum, lr_sss.Y.shape, lr_sss.graph_modes)


    results = {}
    # num_parameter = horpca.num_parameters()
    # results['num_parameter_X'] = num_parameter['X']
    # results['num_parameter_S'] = num_parameter['S']
    # results['num_parameters'] = num_parameter['X'] + num_parameter['S']
    # # diff = len(horpca.Y.ravel()) - results['num_parameters']
    # # if diff <=0:
    #     results['gcv'] = (torch.linalg.norm(horpca.Y - X - S)/ 1e-16)**2
    # else:
    #     results['gcv'] = (torch.linalg.norm(horpca.Y - X - S)/ diff)**2
    
    # results['gcv'] = results['gcv'].item()
    scoring_scores = mchannel.anomaly_scoring_score(lr_sss.S.abs())

    V_sum = lr_sss.V.sum(dim=0, keepdim=False).coalesce().to_dense()
    labels = tensorize(V_sum, lr_sss.Y.shape, lr_sss.graph_modes)!=0

    detection_scores = mchannel.anomaly_detection_score(labels)
    likelihood_scores, likelihood_detection = likelihood_scoring(lr_sss, G, sig=0.999)
    results['au_prc'] = scoring_scores['au_prc']
    results['au_roc'] = scoring_scores['au_roc']
    results['raw_f1'] = detection_scores['f1']
    results['PA_f1'] = detection_scores['PA_F1']

    results['likelihood_au_prc'] = likelihood_scores['au_prc']
    results['likelihood_au_roc'] = likelihood_scores['au_roc']
    results['likelihood_raw_f1'] = likelihood_detection['f1']
    results['likelihood_PA_f1'] = likelihood_detection['PA_F1']
    trial.set_user_attr('raw_f1', results['raw_f1'])
    trial.set_user_attr('roc_auc', results['au_roc'])
    trial.set_user_attr('PA_f1', results['PA_f1'])
    trial.set_user_attr('au_prc', results['au_prc'])

    trial.set_user_attr('likelihood_raw_f1', results['likelihood_raw_f1'])
    trial.set_user_attr('likelihood_roc_auc', results['likelihood_au_roc'])
    trial.set_user_attr('likelihood_PA_f1', results['likelihood_PA_f1'])
    trial.set_user_attr('likelihood_au_prc', results['likelihood_au_prc'])



    
    
    

    for key, value in results.items():
        trial.set_user_attr(key, value)
    # trial.set_user_attr("metrics", results)
    return results['likelihood_au_roc'], results['likelihood_raw_f1']


    
# study = optuna.study.create_study(study_name=f'smd_horpca_M{mchannel.machine_id}_Ch{mchannel.channel_id}_gcv_ow',
#                                   direction='minimize', storage='sqlite:///smd_horpca.db',
#                                   load_if_exists=True)

study.optimize(lrsss_objective, n_trials=500, show_progress_bar=True)
best_trial = max(study.best_trials, key=lambda x: x.values[1])
raw_f1 = best_trial.user_attrs['raw_f1']
PA_f1 = best_trial.user_attrs['PA_f1']
au_roc = best_trial.user_attrs['au_roc'].item()
au_prc = best_trial.user_attrs['au_prc'].item()

likelihood_raw_f1 = best_trial.user_attrs['likelihood_raw_f1']
likelihood_PA_f1 = best_trial.user_attrs['likelihood_PA_f1']
likelihood_au_roc = best_trial.user_attrs['likelihood_au_roc'].item()
likelihood_au_prc = best_trial.user_attrs['likelihood_au_prc'].item()


res = []
res.append({'machine_id': machine_id, 'channel_id': channel_id, 
            'raw_f1': raw_f1, 'likelihood_raw_f1': likelihood_raw_f1, 
            'PA_f1': PA_f1, 'likelihood_PA_f1': likelihood_PA_f1, 
            'au_roc': au_roc, 'likelihood_au_roc': likelihood_au_roc})
df = pd.DataFrame(res)
path_out = '/mnt/home/mondalra/Anomaly_Detection/smd_anomaly_detection/Results/'
filename = path_out+'LRSSS_m'+str(machine_id)+'_c'+str(channel_id)+'.csv'
df.to_csv(filename, index=False)