import os, sys
from time import time
import pickle, argparse
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr

from dist_shap import DistShap
from fast_dist_shap import *
from data import load_reg_data_for_time_comp, load_clf_data_for_time_comp

def run_time_comparison_clf(run_id, sample_size, dimension, DShapley_flag, save_path):
    '''
    This function is the main function to compare elasped time in classification settings
    '''
    # Set directorie and random seed
    directory = save_path+f'/time_comp_clf/{sample_size}_{dimension}/run{run_id}'
    if not os.path.exists(save_path+f'/time_comp_clf/{sample_size}_{dimension}'):
        os.makedirs(save_path+f'/time_comp_clf/{sample_size}_{dimension}')  
    np.random.seed(run_id)

    start_time = time()
    print('-'*30)
    print('FASTDIST')
    print('-'*30)
    (X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_clf_data_for_time_comp(sample_size=sample_size, dimension=dimension)
    clf = LogisticRegression(random_state=0)
    clf.fit(X_dist, y_dist)
    logistic_acc = clf.score(X_train, y_train)

    if dimension < 100:
        utility_minimum_samples = 50
    else:
        utility_minimum_samples = dimension + 100

    beta_dist = np.concatenate((clf.coef_.reshape(-1), clf.intercept_))
    X_dist_tilde, z_dist_tilde, pi_dist, beta_dist = transform_IRLS(X_dist, y_dist, beta=beta_dist)
    X_train_tilde, z_train_tilde, pi_train, beta_dist = transform_IRLS(X_train, y_train, beta=beta_dist)
    raw_data = {'X_dist':X_dist_tilde,
                 'y_dist':z_dist_tilde,
                 'X_star':X_train_tilde,
                 'y_star':z_train_tilde}

    DSV_list = estimate_DSV_ridge(raw_data, utility_minimum_samples=utility_minimum_samples, gamma=0., is_upper=False)
    end_time = time() 
    fastdshap_time = end_time - start_time
    print(f'Elapsed time for FAST DSHAPLEY : {fastdshap_time:.3f}') 

    print('-'*30)
    print('D-Shapley')
    print('-'*30)
    # DShapley
    dshap = DistShap(X=X_train, y=y_train,
                     X_test=X_test, y_test=y_test, num_test=int(len(y_test)//2),
                     X_tot=X_dist, y_tot=y_dist,
                     sources=None,
                     sample_weight=None,
                     model_family='logistic',
                     metric='accuracy',
                     overwrite=False,
                     directory=directory)

    if DShapley_flag is True:
        dshap.run(tmc_run=False, 
             dist_run=True,
             truncation=len(X_train), 
             alpha=None, 
             save_every=100, 
             err=0.05, 
             max_iters=1000)

    vals_fastdist = DSV_list
    vals_dist = np.mean(dshap.results['mem_dist'], 0)

    try:
        dict_results={'time':[dshap.time_dist_run, fastdshap_time],
                      'pearson':pearsonr(vals_fastdist, vals_dist)}
    except:
        dict_results={'time':[0., fastdshap_time],}

    with open(save_path + f'/time_comp_clf/{sample_size}_{dimension}/run_id_{run_id}.pkl', 'wb') as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_time_comparison_reg(run_id, sample_size, dimension, DShapley_flag, save_path):
    '''
    This function is the main function to compare elasped time in regression settings
    '''
    # Set directory and random seed
    directory = save_path+f'/time_comp_reg/{sample_size}_{dimension}/run{run_id}'
    if not os.path.exists(save_path+f'/time_comp_reg/{sample_size}_{dimension}'):
        os.makedirs(save_path+f'/time_comp_reg/{sample_size}_{dimension}')  

    np.random.seed(run_id)
    start_time = time()
    print('-'*30)
    print('FASTDIST')
    print('-'*30)
    (X_dist, y_dist), (X_train, y_train), (X_test, y_test) = load_reg_data_for_time_comp(sample_size=sample_size, dimension=dimension)
    raw_data = {'X_dist':X_dist,
                 'y_dist':y_dist,
                 'X_star':X_train,
                 'y_star':y_train}

    if dimension < 100:
        utility_minimum_samples = 50
    else:
        utility_minimum_samples = dimension + 100
    DSV_list = estimate_DSV_linear(raw_data, utility_minimum_samples=utility_minimum_samples)
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

    if DShapley_flag is True:
        dshap.run(tmc_run=False, 
                 dist_run=True,
                 truncation=len(X_train), 
                 alpha=None, 
                 save_every=100, 
                 err=0.05,
                 max_iters=1000)

    vals_fastdist = DSV_list
    vals_dist = np.mean(dshap.results['mem_dist'], 0)

    try:
        dict_results={'time':[dshap.time_dist_run, fastdshap_time],
                      'pearson':pearsonr(vals_fastdist, vals_dist)}
    except:
        dict_results={'time':[0., fastdshap_time]}

    with open(save_path+f'/time_comp_reg/{sample_size}_{dimension}/run_id_{run_id}.pkl', 'wb') as handle:
        pickle.dump(dict_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--task", type=str, default='reg', choices=['reg', 'clf'])
    parser.add_argument('--DShapley_flag', dest='DShapley_flag', action='store_true')
    parser.set_defaults(DShapley_flag=False)
    parser.add_argument("--save_path", type=str, default='./results')
    args = parser.parse_args()
    run_id, sample_size, dimension = args.run_id, args.sample_size, args.dimension
    task, DShapley_flag, save_path = args.task, args.DShapley_flag, args.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path+'/time_comp_reg')
        os.mkdir(save_path+'/time_comp_clf')

    if task == 'reg':
        print('Time comparison experiment in regression settings')
        run_time_comparison_reg(run_id, sample_size, dimension, DShapley_flag, save_path)
    elif task == 'clf':
        print('Time comparison experiment in classification settings')
        run_time_comparison_clf(run_id, sample_size, dimension, DShapley_flag, save_path)
    else:
        assert False, f'Check task: {task}'




