import os, sys, time
import pickle, argparse
import numpy as np

from DistShap import DistShap
from fastdist import estimate_DSV_linear, estimate_DSV_ridge
from fastdist_utils import print_rank_correlation, point_removal_classification
from data import load_regression_data
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--dataset", 
                    type=str,
                    default='gaussian1', 
                    choices=['gaussian1','gaussian2','whitewine','abalone',
                            'airfoil','diabetes','boston','redwine'])
parser.add_argument("--which_bound", 
                    type=str,
                    default='exact', 
                    choices=['exact','lower','upper'])
args = parser.parse_args()
run_id = args.run_id
dataset = args.dataset
which_bound = args.which_bound
SAVE_PATH = '/oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/temp3'

# Set directorie and random seed
directory = SAVE_PATH + f'/regression/{which_bound}/{dataset}/run{run_id}'
if not os.path.exists(SAVE_PATH+f'/regression/{which_bound}/{dataset}'):
    os.makedirs(SAVE_PATH+f'/regression/{which_bound}/{dataset}')  
np.random.seed(run_id)

if dataset in ['gaussian1','gaussian2']:             
    utility_minimum_samples = 50
elif dataset in ['whitewine','boston','redwine','diabetes','abalone']:             
    utility_minimum_samples = 20
elif dataset in ['airfoil']:             
    utility_minimum_samples = 10
else:
    utility_minimum_samples = 10

print('-'*30)
print('FASTDIST')
print('-'*30)
start_time = time.time()
(X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_regression_data(dataset=dataset)
raw_data = {'X_dist':X_dist,
             'y_dist':y_dist,
             'X_star':X_train,
             'y_star':y_train}

if which_bound == 'exact':
    print('Estimate DSV')
    DSV_list = estimate_DSV_linear(raw_data, utility_minimum_samples=utility_minimum_samples)
elif which_bound == 'upper':
    print('Compute upper bound')
    DSV_list = estimate_DSV_ridge(raw_data, utility_minimum_samples=utility_minimum_samples, is_upper=True)
elif which_bound == 'lower':
    print('Compute lower bound')
    DSV_list = estimate_DSV_ridge(raw_data, utility_minimum_samples=utility_minimum_samples, is_upper=False)
else:
    assert False, f'Check bound options: {which_bound}'
end_time = time.time() 
fastdshap_time = end_time - start_time
print(f'Elapsed time for FAST DSHAPLEY : {fastdshap_time:.3f}') 

y_dist, y_train, y_test = y_dist.reshape(-1), y_train.reshape(-1), y_test.reshape(-1)
reg_model = LinearRegression()
reg_model.fit(X_dist, y_dist)
sigma_2 = np.sum((y_dist - reg_model.predict(X_dist))**2)/(X_dist.shape[0]-X_dist.shape[1])
print(f'Sigma_2 estimates: {sigma_2:.4f}')

print('-'*30)
print('D-Shapley & TMC-Shapley')
print('-'*30)
# DShapley and TMC-Shapley
dshap = DistShap(X=X_train, y=y_train,
                 X_test=X_test, y_test=y_test, num_test=int(len(y_test)//2),
                 X_tot=X_dist, y_tot=y_dist,
                 sources=None,
                 sample_weight=None,
                 model_family='linear',
                 metric='l2',
                 overwrite=False,
                 directory=directory,
                 sigma_2=sigma_2,
                 utility_minimum_samples=utility_minimum_samples)

dshap.run(tmc_run=True, 
         dist_run=True,
         truncation=len(X_train), 
         alpha=None, 
         save_every=100, 
         err=0.01, # 0.05 in paper
         max_iters=1000)

print('-'*30)
print('heldout size','heldout size','test size')
print(len(dshap.X_heldout), len(dshap.y_heldout), len(dshap.y_test))
print('-'*30)

vals_fastdist = DSV_list
vals_tmc = np.mean(dshap.results['mem_tmc'], 0)
vals_dist = np.mean(dshap.results['mem_dist'], 0)

print_rank_correlation(vals_tmc, vals_dist, vals_fastdist)

print('-'*30)
print('Point addition experiment')
print('-'*30)
from shap_utils import portion_performance
n_init = 100
X_new, y_new = dshap.X[n_init:], dshap.y[n_init:]
vals_tmc, vals_dist, vals_fastdist = vals_tmc[n_init:], vals_dist[n_init:], vals_fastdist[n_init:]
X_init, y_init = dshap.X[:n_init], dshap.y[:n_init]
performance_points = np.arange(0, len(X_new)//2, len(X_new)//40)
x_sqn = performance_points / len(X_new) * 100

perf_func = lambda order: portion_performance(dshap, order, performance_points,
                                                 X_new, y_new, X_init, y_init,
                                                  dshap.X_heldout, dshap.y_heldout)

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

with open(SAVE_PATH + f'/regression/{which_bound}/{dataset}/run_id_{run_id}.pkl', 'wb') as handle:
    pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)





