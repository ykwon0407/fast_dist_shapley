import os, sys, time
import pickle, argparse
import numpy as np

from DistShap import DistShap
from fastdist import estimate_DSV_linear
from fastdist_utils import print_rank_correlation
from data import load_time_comparison_data
from sklearn.linear_model import LinearRegression
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
SAVE_PATH = '/oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/temp4'

# Set directorie and random seed
directory = SAVE_PATH+f'/time_regression/{dataset}_{sample_size}_{dimension}/run{run_id}'
if not os.path.exists(SAVE_PATH+f'/time_regression/{dataset}_{sample_size}_{dimension}'):
    os.makedirs(SAVE_PATH+f'/time_regression/{dataset}_{sample_size}_{dimension}')  
np.random.seed(run_id)

start_time = time.time()
print('-'*30)
print('FASTDIST')
print('-'*30)
(X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_time_comparison_data(sample_size=sample_size, dimension=dimension)
raw_data = {'X_dist':X_dist,
             'y_dist':y_dist,
             'X_star':X_train,
             'y_star':y_train}

if dimension < 100:
    utility_minimum_samples = 50
else:
    utility_minimum_samples = dimension + 100
DSV_list = estimate_DSV_linear(raw_data, utility_minimum_samples=utility_minimum_samples)
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

if is_DShapley == True:
    dshap.run(tmc_run=False, 
             dist_run=True,
             truncation=len(X_train), 
             alpha=None, 
             save_every=100, 
             err=0.05, # 0.05 in paper
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

with open(SAVE_PATH + f'/time_regression/{dataset}_{sample_size}_{dimension}/run_id_{run_id}.pkl', 'wb') as handle:
    pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)





