import os, sys, time
import pickle, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression

from DistShap import DistShap
from fastdist import estimate_DSV_ridge
from fastdist_utils import transform_IRLS, print_rank_correlation, point_removal_classification
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
parser.add_argument("--specific_class", type=int, default=1)
parser.add_argument('--is_upper', dest='is_upper', action='store_true')
parser.set_defaults(is_upper=False)
args = parser.parse_args()
run_id = args.run_id
dataset = args.dataset
specific_class = args.specific_class
is_upper = args.is_upper
SAVE_PATH = '/oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/temp3'

which_bound = 'upper' if is_upper is True else 'lower' 
print(f'which bound? : {which_bound}')

# Set directorie and random seed
directory = SAVE_PATH + f'/classification/{which_bound}/{dataset}/run{run_id}'
if not os.path.exists(SAVE_PATH+f'/classification/{which_bound}/{dataset}'):
    os.makedirs(SAVE_PATH+f'/classification/{which_bound}/{dataset}')  
np.random.seed(run_id)

print('-'*30)
print('FASTDIST')
print('-'*30)
start_time = time.time()
(X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_classification_data(dataset=dataset,
                                                                                 specific_class=specific_class)

# Logistic regression estimator
clf = LogisticRegression(random_state=0)
clf.fit(X_dist, y_dist)
logistic_acc = clf.score(X_train, y_train)

if dataset not in ['covertype', 'cifar10', 'australian_scale']:
    X_dist_tilde, z_dist_tilde, pi_dist, beta_dist = transform_IRLS(X_dist, y_dist, beta=None) # classification
else:
    beta_dist = np.concatenate((clf.coef_.reshape(-1), clf.intercept_))
    X_dist_tilde, z_dist_tilde, pi_dist, beta_dist = transform_IRLS(X_dist, y_dist, beta=beta_dist) # classification
X_train_tilde, z_train_tilde, pi_train, beta_dist = transform_IRLS(X_train, y_train, beta=beta_dist)
raw_data = {'X_dist':X_dist_tilde,
             'y_dist':z_dist_tilde,
             'X_star':X_train_tilde,
             'y_star':z_train_tilde}

if dataset in ['cifar10', 'mnist', 'fashion']:             
    utility_minimum_samples = 50 
elif dataset in ['covertype']:             
    utility_minimum_samples = 75
elif dataset in ['diabetes_scale', 'australian_scale']:
    utility_minimum_samples = 15
else:
    utility_minimum_samples = 10

DSV_list = estimate_DSV_ridge(raw_data, utility_minimum_samples=utility_minimum_samples, gamma=0., is_upper=is_upper)
end_time = time.time() 
y_train_pred = (pi_train > 0.5) + 0.0
glm_acc = np.mean(y_train_pred == y_train)
del raw_data, X_dist_tilde, z_dist_tilde, X_train_tilde, z_train_tilde
fastdshap_time = end_time - start_time
print(f'Elapsed time for FAST DSHAPLEY : {fastdshap_time:.3f}') 

print(f'GLM coef: {beta_dist}')
print(f'GLM accuracy: {glm_acc}')
print(f'Logistic coef: {clf.coef_},{clf.intercept_}')
print(f'Logistic accuracy: {logistic_acc}')

print('-'*30)
print('D-Shapley & TMC-Shapley')
print('-'*30)
# DShapley and TMC-Shapley
dshap = DistShap(X=X_train, y=y_train,
                 X_test=X_test, y_test=y_test, num_test=int(len(y_test)//2),
                 X_tot=X_dist, y_tot=y_dist,
                 sources=None,
                 sample_weight=None,
                 model_family='logistic',
                 metric='accuracy',
                 overwrite=False,
                 directory=directory)

dshap.run(tmc_run=False, 
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
if dataset in ['gaussian1', 'gaussian2']:
    X_init, y_init = X_init[:10], y_init[:10]
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
'base_results':[logistic_acc, glm_acc],
'time':[dshap.time_dist_run, fastdshap_time, dshap.time_tmc_run],
'dist': [d_perf_inc / d_perf_inc[0] * 100, d_perf_dec / d_perf_dec[0] * 100],
'fastdist': [fastd_perf_inc / fastd_perf_inc[0] * 100, fastd_perf_dec / fastd_perf_dec[0] * 100],
'data': [tmc_perf_inc / tmc_perf_inc[0] * 100, tmc_perf_dec / tmc_perf_dec[0] * 100],
'rnd': [rnd_perf_inc / rnd_perf_inc[0] * 100, rnd_perf_dec / rnd_perf_dec[0] * 100],
'spear':spearmanr(vals_fastdist, vals_dist),
'pearson':pearsonr(vals_fastdist, vals_dist),
}

with open(SAVE_PATH + f'/classification/{which_bound}/{dataset}/run_id_{run_id}.pkl', 'wb') as handle:
    pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)





