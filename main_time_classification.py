import os, sys, time
import pickle, argparse
import numpy as np

from DistShap import DistShap
from fastdist import estimate_DSV_ridge
from data import load_time_comparison_clf_data
from fastdist_utils import transform_IRLS
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=0)
parser.add_argument("--sample_size", type=int, default=200)
parser.add_argument("--dimension", type=int, default=10)
parser.add_argument('--is_DShapley', dest='is_DShapley', action='store_true')
parser.set_defaults(is_DShapley=False)
args = parser.parse_args()
run_id = args.run_id
sample_size, dimension = args.sample_size, args.dimension
is_DShapley = args.is_DShapley
dataset='gaussian'
SAVE_PATH = '/oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/temp3'

# Set directorie and random seed
directory = SAVE_PATH+f'/time_classification/{dataset}_{sample_size}_{dimension}/run{run_id}'
if not os.path.exists(SAVE_PATH+f'/time_classification/{dataset}_{sample_size}_{dimension}'):
    os.makedirs(SAVE_PATH+f'/time_classification/{dataset}_{sample_size}_{dimension}')  
np.random.seed(run_id)

start_time = time.time()
print('-'*30)
print('FASTDIST')
print('-'*30)
(X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_time_comparison_clf_data(sample_size=sample_size, dimension=dimension)

clf = LogisticRegression(random_state=0)
clf.fit(X_dist, y_dist)
logistic_acc = clf.score(X_train, y_train)

if dimension < 100:
    utility_minimum_samples = 50
else:
    utility_minimum_samples = dimension + 100

beta_dist = np.concatenate((clf.coef_.reshape(-1), clf.intercept_))
X_dist_tilde, z_dist_tilde, pi_dist, beta_dist = transform_IRLS(X_dist, y_dist, beta=beta_dist) # classification
X_train_tilde, z_train_tilde, pi_train, beta_dist = transform_IRLS(X_train, y_train, beta=beta_dist)
raw_data = {'X_dist':X_dist_tilde,
             'y_dist':z_dist_tilde,
             'X_star':X_train_tilde,
             'y_star':z_train_tilde}

DSV_list = estimate_DSV_ridge(raw_data, utility_minimum_samples=utility_minimum_samples, gamma=0., is_upper=False)
end_time = time.time() 
fastdshap_time = end_time - start_time
print(f'Elapsed time for FAST DSHAPLEY : {fastdshap_time:.3f}') 

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

if is_DShapley == True:
    dshap.run(tmc_run=False, 
         dist_run=True,
         truncation=len(X_train), 
         alpha=None, 
         save_every=100, 
         err=0.01, # 0.05 in paper
         max_iters=1000)

vals_fastdist = DSV_list
vals_tmc = np.mean(dshap.results['mem_tmc'], 0)
vals_dist = np.mean(dshap.results['mem_dist'], 0)

try:
    dict_results={
    'time':[dshap.time_dist_run, fastdshap_time, dshap.time_tmc_run],
    'spear':spearmanr(vals_fastdist, vals_dist),
    'pearson':pearsonr(vals_fastdist, vals_dist),
    }
except:
    dict_results={
    'time':[0., fastdshap_time, 0.],
    }

with open(SAVE_PATH + f'/time_classification/{dataset}_{sample_size}_{dimension}/run_id_{run_id}.pkl', 'wb') as handle:
    pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)





