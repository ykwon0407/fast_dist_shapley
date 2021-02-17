import os, sys
from time import time
import pickle, argparse
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

from dist_shap import DistShap, DistShapDensity
from shap_utils import portion_performance
from fast_dist_shap import *
from data import load_reg_data_for_point_addition, load_non_reg_data_for_point_addition

def run_point_addition_reg(run_id, dataset, which_bound, save_path):
    # Set directorie and random seed
    if not os.path.exists(save_path+f'/regression/{which_bound}/{dataset}'):
        os.makedirs(save_path+f'/regression/{which_bound}/{dataset}')  
    directory = save_path + f'/regression/{which_bound}/{dataset}/run{run_id}'
    np.random.seed(run_id)
    if dataset in ['gaussian']:             
        utility_minimum_samples = 50
    elif dataset in ['whitewine','abalone']:             
        utility_minimum_samples = 20
    elif dataset in ['airfoil']:             
        utility_minimum_samples = 10
    else:
        assert False, f'Check {dataset}'

    print('-'*30)
    print('FASTDIST')
    print('-'*30)
    start_time = time()
    (X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_reg_data_for_point_addition(dataset=dataset)
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
    end_time = time() 
    fastdshap_time = end_time - start_time
    print(f'Elapsed time for FAST DSHAPLEY : {fastdshap_time:.3f}') 

    y_dist, y_train, y_test = y_dist.reshape(-1), y_train.reshape(-1), y_test.reshape(-1)
    reg_model = LinearRegression()
    reg_model.fit(X_dist, y_dist)
    sigma_2 = np.sum((y_dist - reg_model.predict(X_dist))**2)/(X_dist.shape[0]-X_dist.shape[1])
    print(f'Sigma_2 estimates: {sigma_2:.4f}')

    print('-'*30)
    print('D-Shapley')
    print('-'*30)
    # DShapley
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

    dshap.run(tmc_run=False, 
             dist_run=True,
             truncation=len(X_train), 
             alpha=None, 
             save_every=100, 
             err=0.05, 
             max_iters=1000)

    print('-'*30)
    print('heldout size','heldout size','test size')
    print(len(dshap.X_heldout), len(dshap.y_heldout), len(dshap.y_test))
    print('-'*30)

    vals_fastdist = DSV_list
    vals_dist = np.mean(dshap.results['mem_dist'], 0)
    print_rank_correlation(vals_dist, vals_fastdist)

    print('-'*30)
    print('Point addition experiment')
    print('-'*30)
    n_init = 100
    X_new, y_new = dshap.X[n_init:], dshap.y[n_init:]
    vals_dist, vals_fastdist = vals_dist[n_init:], vals_fastdist[n_init:]
    X_init, y_init = dshap.X[:n_init], dshap.y[:n_init]
    performance_points = np.arange(0, len(X_new)//2, len(X_new)//40)
    x_sqn = performance_points / len(X_new) * 100
    perf_func = lambda order: portion_performance(dshap, order, performance_points,
                                                     X_new, y_new, X_init, y_init,
                                                      dshap.X_heldout, dshap.y_heldout)

    # From smallest to largest
    fastd_perf_inc = perf_func(np.argsort(-vals_fastdist))
    d_perf_inc = perf_func(np.argsort(-vals_dist)) # np.argsort(-vals_dist_new) is decreasing.
    rnd_perf_inc = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

    # From largest to smallest
    d_perf_dec = perf_func(np.argsort(vals_dist))
    fastd_perf_dec = perf_func(np.argsort(vals_fastdist))
    rnd_perf_dec = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

    dict_results={'x_sqn':x_sqn,
                  'time':[dshap.time_dist_run, fastdshap_time],
                  'dist': [d_perf_inc / d_perf_inc[0] * 100, d_perf_dec / d_perf_dec[0] * 100],
                  'fastdist': [fastd_perf_inc / fastd_perf_inc[0] * 100, fastd_perf_dec / fastd_perf_dec[0] * 100],
                  'rnd': [rnd_perf_inc / rnd_perf_inc[0] * 100, rnd_perf_dec / rnd_perf_dec[0] * 100],
                  'pearson':pearsonr(vals_fastdist, vals_dist)}

    with open(save_path + f'/regression/{which_bound}/{dataset}/run_id_{run_id}.pkl', 'wb') as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_point_addition_clf(run_id, dataset, specific_class, which_bound, save_path):
    # Set directorie and random seed
    directory = save_path + f'/classification/{which_bound}/{dataset}/run{run_id}'
    if not os.path.exists(save_path+f'/classification/{which_bound}/{dataset}'):
        os.makedirs(save_path+f'/classification/{which_bound}/{dataset}')  
    np.random.seed(run_id)

    print('-'*30)
    print('FASTDIST')
    print('-'*30)
    start_time = time()
    (X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_non_reg_data_for_point_addition(dataset=dataset,
                                                                                     specific_class=specific_class)

    # Logistic regression estimator
    clf = LogisticRegression(random_state=0)
    clf.fit(X_dist, y_dist)
    logistic_acc = clf.score(X_train, y_train)

    if dataset not in ['cifar10']:
        X_dist_tilde, z_dist_tilde, pi_dist, beta_dist = transform_IRLS(X_dist, y_dist, beta=None) # classification
    else:
        beta_dist = np.concatenate((clf.coef_.reshape(-1), clf.intercept_))
        X_dist_tilde, z_dist_tilde, pi_dist, beta_dist = transform_IRLS(X_dist, y_dist, beta=beta_dist) # classification
    X_train_tilde, z_train_tilde, pi_train, beta_dist = transform_IRLS(X_train, y_train, beta=beta_dist)
    raw_data = {'X_dist':X_dist_tilde,
                 'y_dist':z_dist_tilde,
                 'X_star':X_train_tilde,
                 'y_star':z_train_tilde}

    if dataset in ['cifar10', 'mnist']:             
        utility_minimum_samples = 50 
    elif dataset in ['gaussian', 'skin_nonskin']:
        utility_minimum_samples = 10
    else:
        assert False, f'Check {dataset}'

    DSV_list = estimate_DSV_ridge(raw_data, utility_minimum_samples=utility_minimum_samples, gamma=0., is_upper=is_upper)
    end_time = time() 
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
             err=0.05,
             max_iters=1000)

    print('-'*30)
    print('heldout size','heldout size','test size')
    print(len(dshap.X_heldout), len(dshap.y_heldout), len(dshap.y_test))
    print('-'*30)

    vals_fastdist = DSV_list
    vals_dist = np.mean(dshap.results['mem_dist'], 0)
    print_rank_correlation(vals_dist, vals_fastdist)

    print('-'*30)
    print('Point addition experiment')
    print('-'*30)
    from shap_utils import portion_performance
    n_init = 100

    X_new, y_new = dshap.X[n_init:], dshap.y[n_init:]
    vals_dist, vals_fastdist = vals_dist[n_init:], vals_fastdist[n_init:]
    X_init, y_init = dshap.X[:n_init], dshap.y[:n_init]
    if dataset in ['gaussian']:
        X_init, y_init = X_init[:10], y_init[:10]
    performance_points = np.arange(0, len(X_new)//2, len(X_new)//40)
    x_sqn = performance_points / len(X_new) * 100

    perf_func = lambda order: portion_performance(dshap, order, performance_points,
                                                     X_new, y_new, X_init, y_init,
                                                      dshap.X_heldout, dshap.y_heldout)

    # From smallest to largest
    fastd_perf_inc = perf_func(np.argsort(-vals_fastdist))
    d_perf_inc = perf_func(np.argsort(-vals_dist)) # np.argsort(-vals_dist_new) is decreasing.
    rnd_perf_inc = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

    # From largest to smallest
    d_perf_dec = perf_func(np.argsort(vals_dist))
    fastd_perf_dec = perf_func(np.argsort(vals_fastdist))
    rnd_perf_dec = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

    dict_results={
    'x_sqn':x_sqn,
    'base_results':[logistic_acc, glm_acc],
    'time':[dshap.time_dist_run, fastdshap_time],
    'dist': [d_perf_inc / d_perf_inc[0] * 100, d_perf_dec / d_perf_dec[0] * 100],
    'fastdist': [fastd_perf_inc / fastd_perf_inc[0] * 100, fastd_perf_dec / fastd_perf_dec[0] * 100],
    'rnd': [rnd_perf_inc / rnd_perf_inc[0] * 100, rnd_perf_dec / rnd_perf_dec[0] * 100],
    'pearson':pearsonr(vals_fastdist, vals_dist),
    }

    with open(save_path + f'/classification/{which_bound}/{dataset}/run_id_{run_id}.pkl', 'wb') as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_point_addition_density(run_id, dataset, save_path):
    # Set directorie and random seed
    directory = save_path+f'/density/{dataset}/run{run_id}'
    if not os.path.exists(save_path+f'/density/{dataset}'):
        os.makedirs(save_path+f'/density/{dataset}')  
    np.random.seed(run_id)

    print('-'*30)
    print('FASTDIST')
    print('-'*30)
    start_time = time()
    # Load data points 
    (X_dist, _), (X_train, _), (X_test, _) = load_non_reg_data_for_point_addition(dataset=dataset)
    X_dist = X_dist[:2000]

    # Find the best bandwidth 
    params = {'bandwidth': np.logspace(-2, 1, 7)}
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
    grid.fit(X_dist)
    kde = grid.best_estimator_
    optimal_bandwidth = kde.bandwidth
    kde = grid.best_estimator_
    num_test = len(X_test)//2
    X_from_estimator = kde.sample(num_test) # sample from the density estimator
    risk = np.mean(np.exp(kde.score_samples(X_from_estimator)))-2*np.mean(np.exp(kde.score_samples((X_test[-num_test:])))) # Utility computation
    print('-'*30)
    print(f"best bandwidth: {optimal_bandwidth:.3f}")
    print(f"best risk: {risk}")
    print('-'*30)

    # DShapley
    raw_data = {'X_dist':X_dist,
                 'X_star':X_train,
                 'bandwidth':optimal_bandwidth}
    DSV_list = estimate_DSV_density(raw_data)
    end_time = time() 
    fastdshap_time = end_time - start_time
    print(f'Elapsed time for FAST DSHAPLEY : {fastdshap_time:.3f}') 

    print('-'*30)
    print('D-Shapley')
    print('-'*30)
    # DShapley 
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
             err=0.05,
             max_iters=100)

    print('-'*30)
    print('heldout size', 'test size')
    print(len(dshap.X_heldout), len(dshap.X_test))
    print('-'*30)

    vals_fastdist = DSV_list
    vals_dist = np.mean(dshap.results['mem_dist'], 0)
    print_rank_correlation(vals_dist, vals_fastdist)

    print('-'*30)
    print('Point addition experiment')
    print('-'*30)
    from shap_utils import portion_performance_density
    n_init = 100

    X_new = dshap.X[n_init:]
    vals_dist, vals_fastdist = vals_dist[n_init:], vals_fastdist[n_init:]
    X_init = dshap.X[:n_init]
    performance_points = np.arange(0, len(X_new)//2, len(X_new)//40)
    x_sqn = performance_points / len(X_new) * 100

    perf_func = lambda order: portion_performance_density(dshap, order, performance_points, X_new, X_init, dshap.X_heldout)

    # From smallest to largest
    fastd_perf_inc = perf_func(np.argsort(-vals_fastdist))
    d_perf_inc = perf_func(np.argsort(-vals_dist)) # np.argsort(-vals_dist_new) is decreasing.
    rnd_perf_inc = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

    # From largest to smallest
    d_perf_dec = perf_func(np.argsort(vals_dist))
    fastd_perf_dec = perf_func(np.argsort(vals_fastdist))
    rnd_perf_dec = np.mean([perf_func(np.random.permutation(len(vals_fastdist))) for _ in range(10)], 0)

    dict_results={
    'x_sqn':x_sqn,
    'time':[dshap.time_dist_run, fastdshap_time],
    'dist': [d_perf_inc / d_perf_inc[0] * 100, d_perf_dec / d_perf_dec[0] * 100],
    'fastdist': [fastd_perf_inc / fastd_perf_inc[0] * 100, fastd_perf_dec / fastd_perf_dec[0] * 100],
    'rnd': [rnd_perf_inc / rnd_perf_inc[0] * 100, rnd_perf_dec / rnd_perf_dec[0] * 100],
    'pearson':pearsonr(vals_fastdist, vals_dist),
    }

    with open(save_path + f'/density/{dataset}/run_id_{run_id}.pkl', 'wb') as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='gaussian')
    parser.add_argument("--task", type=str, default='reg', choices=['reg', 'clf', 'density'])
    parser.add_argument("--which_bound", type=str, default='exact')
    parser.add_argument("--save_path", type=str, default='./results')
    parser.add_argument("--specific_class", type=int, default=1)
    parser.add_argument('--is_upper', dest='is_upper', action='store_true')
    parser.set_defaults(is_upper=False)
    args = parser.parse_args()
    run_id, dataset, task = args.run_id, args.dataset, args.task
    which_bound, save_path = args.which_bound, args.save_path
    specific_class, is_upper = args.specific_class, args.is_upper

    if not os.path.exists(save_path+'/regression'):
        os.makedirs(save_path+'/regression')
        os.makedirs(save_path+'/classification')
        os.makedirs(save_path+'/density')
    
    if task == 'reg':
        print('Point addition experiment in regression settings')
        run_point_addition_reg(run_id, dataset, which_bound, save_path)
    elif task == 'clf':
        print('Point addition experiment in classification settings')
        which_bound = 'upper' if is_upper is True else 'lower' 
        print(f'which bound? : {which_bound}')
        run_point_addition_clf(run_id, dataset, specific_class, which_bound, save_path)
    elif task == 'density':
        print('Point addition experiment in density estimation problems')
        run_point_addition_density(run_id, dataset, save_path)
    else:
        assert False, f'Check task: {task}'


