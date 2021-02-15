import os, sys, time
import pickle, argparse
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from DistShap import DistShapDensity
from fastdist import estimate_DSV_density
from fastdist_utils import print_rank_correlation, point_removal_classification
from data import load_classification_data
from scipy.stats import spearmanr, pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--dataset", 
                    type=str,
                    default='gaussian1', 
                    choices=['gaussian1', 'gaussian2', 'covertype', 
                            'diabetes_scale','australian_scale',
                            'skin_nonskin', 'cifar10', 'mnist', 'fashion'])
args = parser.parse_args()
run_id = args.run_id
dataset = args.dataset
SAVE_PATH = '/oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/temp3'

# Set directorie and random seed
directory = SAVE_PATH+f'/density/{dataset}/run{run_id}'
if not os.path.exists(SAVE_PATH+f'/density/{dataset}'):
    os.makedirs(SAVE_PATH+f'/density/{dataset}')  
np.random.seed(run_id)

print('-'*30)
print('FASTDIST')
print('-'*30)
start_time = time.time()
# Load data points 
(X_dist, _), (X_train, _), (X_test, _) = load_classification_data(dataset=dataset)
X_dist = X_dist[:2000]

# Find the best bandwidth 
params = {'bandwidth': np.logspace(-2, 1, 7)}
grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
grid.fit(X_dist)
kde = grid.best_estimator_
optimal_bandwidth = kde.bandwidth
kde = grid.best_estimator_
X_from_estimator = kde.sample(len(X_test)) # sample from the density estimator
risk = np.mean(np.exp(kde.score_samples(X_from_estimator)))-2*np.mean(np.exp(kde.score_samples(X_test))) # Utility computation
print('-'*30)
print(f"best bandwidth: {optimal_bandwidth:.3f}")
print(f"best risk: {risk}")
print('-'*30)

# DShapley
raw_data = {'X_dist':X_dist,
             'X_star':X_train,
             'bandwidth':optimal_bandwidth}
DSV_list = estimate_DSV_density(raw_data)
end_time = time.time() 
fastdshap_time = end_time - start_time
print(f'Elapsed time for FAST DSHAPLEY : {fastdshap_time:.3f}') 

print('-'*30)
print('D-Shapley & TMC-Shapley')
print('-'*30)
# DShapley and TMC-Shapley
dshap = DistShapDensity(X=X_train, X_test=X_test, num_test=int(len(X_test)//2),
                         bandwidth=optimal_bandwidth,
                         X_tot=X_dist, 
                         sources=None,
                         sample_weight=None,
                         overwrite=False,
                         directory=directory)

dshap.run(tmc_run=False, 
         dist_run=True,
         truncation=len(X_train), 
         alpha=None, 
         save_every=100, 
         err=0.01, # 0.05 in paper
         max_iters=100)

print('-'*30)
print('heldout size', 'test size')
print(len(dshap.X_heldout), len(dshap.X_test))
print('-'*30)

vals_fastdist = DSV_list
vals_tmc = np.mean(dshap.results['mem_tmc'], 0)
vals_dist = np.mean(dshap.results['mem_dist'], 0)

print_rank_correlation(vals_tmc, vals_dist, vals_fastdist)

print('-'*30)
print('Point addition experiment')
print('-'*30)
from shap_utils import portion_performance_density
n_init = 100

X_new = dshap.X[n_init:]
vals_tmc, vals_dist, vals_fastdist = vals_tmc[n_init:], vals_dist[n_init:], vals_fastdist[n_init:]
X_init = dshap.X[:n_init]
performance_points = np.arange(0, len(X_new)//2, len(X_new)//40)
x_sqn = performance_points / len(X_new) * 100

perf_func = lambda order: portion_performance_density(dshap, order, performance_points, X_new, X_init, dshap.X_heldout)

# From smallest to largest
fastd_perf_inc = perf_func(np.argsort(-vals_fastdist))
d_perf_inc = perf_func(np.argsort(-vals_dist)) # np.argsort(-vals_dist_new) is decreasing.
tmc_perf_inc = perf_func(np.argsort(-vals_tmc))
rnd_perf_inc = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

# From largest to smallest
d_perf_dec = perf_func(np.argsort(vals_dist))
tmc_perf_dec = perf_func(np.argsort(vals_tmc))
fastd_perf_dec = perf_func(np.argsort(vals_fastdist))
rnd_perf_dec = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

dict_results={
'x_sqn':x_sqn,
'time':[dshap.time_dist_run, fastdshap_time, dshap.time_tmc_run],
'dist': [d_perf_inc / d_perf_inc[0] * 100, d_perf_dec / d_perf_dec[0] * 100],
'fastdist': [fastd_perf_inc / fastd_perf_inc[0] * 100, fastd_perf_dec / fastd_perf_dec[0] * 100],
'data': [tmc_perf_inc / tmc_perf_inc[0] * 100, tmc_perf_dec / tmc_perf_dec[0] * 100],
'rnd': [rnd_perf_inc / rnd_perf_inc[0] * 100, rnd_perf_dec / rnd_perf_dec[0] * 100],
'spear':spearmanr(vals_fastdist, vals_dist),
'pearson':pearsonr(vals_fastdist, vals_dist),
}

with open(SAVE_PATH + f'/density/{dataset}/run_id_{run_id}.pkl', 'wb') as handle:
    pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)





