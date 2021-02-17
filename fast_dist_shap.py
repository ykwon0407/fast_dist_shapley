import numpy as np
import pandas as pd
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from scipy.stats import spearmanr
import statsmodels.api as sm

"""
basic functions
"""
def normalize_X(df, mean, std):
    return (df-mean)/(std+1e-12)

def transform_IRLS(X, y, beta=None):
    X_tmp = sm.add_constant(X, prepend=False, has_constant='add')
    if beta is None:
        glm_binom = sm.GLM(y, X_tmp, family=sm.families.Binomial(), maxiter=1000)
        res = glm_binom.fit()
        print(res.summary(), flush=True)
        beta = res.params
    
    eta = X_tmp.dot(beta)
    pi_hat = np.exp(eta)/(1+np.exp(eta))
    weights = pi_hat*(1-pi_hat)
    z = eta + (y - pi_hat)/(weights+1e-16)

    z_tilde = z*(weights**0.5)
    X_tilde = X*((weights**0.5).reshape(-1,1))
    
    return X_tilde, z_tilde, pi_hat, beta

def preprocess_data(data_dict):
    X_dist_tmp, y_dist_tmp = data_dict['X_dist'], data_dict['y_dist']
    X_star_tmp, y_star_tmp = data_dict['X_star'], data_dict['y_star']
    print('Dist (m,p) : ', X_dist_tmp.shape)
    print('to be valued (m,p) : ', X_star_tmp.shape)
    
    # centering part
    X_mean, y_mean = np.mean(X_dist_tmp, axis=0), np.mean(y_dist_tmp)
    X_dist_tmp = X_dist_tmp - X_mean
    X_star_tmp = X_star_tmp - X_mean
    y_dist_tmp = y_dist_tmp - y_mean
    y_star_tmp = y_star_tmp - y_mean
    
    # diagonal_matrix = 1e-8*np.diagflat(np.ones(X_dist_tmp.shape[1]))
    beta_dist = np.linalg.inv(X_dist_tmp.T.dot(X_dist_tmp)).dot(X_dist_tmp.T.dot(y_dist_tmp)) 
    resi_dist = (y_dist_tmp - X_dist_tmp.dot(beta_dist)).reshape(-1) 
    resi_star = (y_star_tmp - X_star_tmp.dot(beta_dist)).reshape(-1) 

    # whitening part
    sample_X_cov = X_dist_tmp.T.dot(X_dist_tmp)/X_dist_tmp.shape[0] # Covariance of X_dist
    d, V = np.linalg.eigh(sample_X_cov)
    D = np.diag(1. / np.sqrt(d+1e-12))
    W = np.dot(np.dot(V, D), V.T) # (Covariance of X_dist ** 0.5)

    # multiply by the whitening matrix
    X_dist_whitened = np.dot(X_dist_tmp, W) 
    X_star_whitened = np.dot(X_star_tmp, W) 
    
    return (X_dist_whitened, y_dist_tmp, resi_dist), (X_star_whitened, y_star_tmp, resi_star), beta_dist


def print_rank_correlation(vals_dist, vals_fastdist):
    print('-'*30)
    print('Rank correlation vs random')
    print('-'*30)

    val_list = [vals_dist, vals_fastdist]
    name_list = ['D-Shapley','FastDist']

    for i in range(len(name_list)): 
        corr = spearmanr(np.random.normal(size=len(vals_fastdist)), val_list[i])[0]
        print(f'Rank correlation: {name_list[i]} vs random values = {corr:.3f}')

    print('-'*30)
    print('Rank correlation')
    print('-'*30)

    for i in range(len(name_list)):
        for j in range(len(name_list)):
            if i < j:
                corr = spearmanr(val_list[i], val_list[j])[0]
                print(f'Rank correlation: {name_list[i]} vs {name_list[j]} = {corr:.3f}')

def point_removal_classification(logistic_model, order, performance_points, X, y, X_heldout, y_heldout):
    error_list = []
    for ind in performance_points:
        current_index = order[ind:]
        X_batch, y_batch = X[current_index], y[current_index]
        try:
            logistic_model.fit(X_batch, y_batch)
            current_error = logistic_model.score(X_heldout, y_heldout)
        except:
            print('Is it unique? ', np.unique(y_batch))
            p_heldout = np.mean(y_heldout)
            current_error = max(p_heldout, 1-p_heldout)
        error_list.append(current_error)    
    return error_list


'''
Linear regression 
'''

TOL = 0.005

def estimate_DSV_linear_core(x_norm, 
                             subset_size, 
                             input_dim, 
                             error_2, 
                             sigma_2, 
                             MC_limit=10000,
                             tol=0.01, 
                             patience_limit=10):
    n_patience_cal, new_exp = 0, 0
    for i in range(MC_limit):
        old_exp = new_exp
        
        # Generate chi-square distribution with the `subset_size-input_dim+1` degree of freedom
        chi_part=np.random.normal(size=(subset_size-input_dim+1,1))
        chi_dist=chi_part.T.dot(chi_part)[0,0] 
        denom = (x_norm+chi_dist)
        numer = (x_norm*error_2+chi_dist*sigma_2)

        new_exp = (old_exp*i + ((subset_size-1)/(subset_size-input_dim))*(numer/(denom**2)))/(i+1)
        relative_diff = np.abs(new_exp-old_exp)/(old_exp+1e-12) 

        if relative_diff < tol:
            n_patience_cal += 1
            if n_patience_cal >= patience_limit:
                break
    return new_exp

def estimate_DSV_linear_from_x_norm(x_norm=1.0,
                                     m=100, 
                                     input_dim=5, 
                                     utility_minimum_samples=10, 
                                     error_2=1, 
                                     sigma_2=1, 
                                     MC_limit=10000, 
                                     tol=TOL, 
                                     patience_limit=10):
    n_patience, nu_part = 0, 0
    for subset_size in range(utility_minimum_samples,m+1):
        expectation_part = estimate_DSV_linear_core(x_norm, subset_size, input_dim, error_2, sigma_2, MC_limit)
        nu_part_diff = -expectation_part/m
        nu_part += nu_part_diff
        if np.abs((nu_part_diff)/(nu_part)) < tol:
            n_patience += 1
            if n_patience >= patience_limit:
                break    
    return nu_part

def estimate_DSV_linear(raw_data, utility_minimum_samples=10):
    (X_dist_whitened, _, resi_dist), (X_star_whitened, _, resi_star), _ = preprocess_data(raw_data)
    sigma_2 = np.mean(resi_dist ** 2)

    DSV_list=[]
    for i in range(X_star_whitened.shape[0]):
        x_norm=sum(X_star_whitened[i]**2) # Assumed that a covariance matrix is the identity matrix.
        dsv=estimate_DSV_linear_from_x_norm(x_norm=x_norm,
                                             m=X_star_whitened.shape[0], 
                                             input_dim=X_star_whitened.shape[1],
                                             utility_minimum_samples=utility_minimum_samples, 
                                             error_2=(resi_star[i]**2),
                                             sigma_2=sigma_2)
        DSV_list.append(dsv)
    DSV_list = np.array(DSV_list)
    return DSV_list


'''
Density Estimation 
'''

def density_const_A(n,m):
    const_A = 0
    for j in range(1,m+1,1):
        const_A += ((n)**2)/((j+n-1)**2) / m
    return const_A

def density_const_B(n,m):
    const_B = 0
    for j in range(2,m+1,1):
        const_B += 2*n*(j-1)/((j+n-1)**2) / m
    return const_B    

def estimate_DSV_density_core(x, m, X_dist, bandwidth=0.1):
    x = x.reshape(1,-1)
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(x)

    random_sample_from_phat = kde.sample(len(X_dist))
    phat_at_random_sample_from_phat = np.exp(kde.score_samples(random_sample_from_phat))
    expected_phat = np.exp(kde.score_samples(X_dist))
    differences = kde.score_samples(x + (random_sample_from_phat - X_dist))

    term_1 = np.mean(phat_at_random_sample_from_phat)-2*np.mean(expected_phat)
    term_2 = np.mean(expected_phat) - np.mean(np.exp(differences))

    dsv = -density_const_A(1,m)*term_1 + density_const_B(1,m)*term_2
    return dsv

def estimate_DSV_density(raw_data):
    X_dist, X_star = raw_data['X_dist'], raw_data['X_star']
    bandwidth = raw_data['bandwidth']
    m=X_star.shape[0]
    DSV_list=[]
    for i in range(m):
        dsv=estimate_DSV_density_core(x=X_star[i], m=m, X_dist=X_dist, bandwidth=bandwidth)
        DSV_list.append(dsv)
    DSV_list = np.array(DSV_list)
    return DSV_list


'''
Ridge bounds 
'''

def estimate_DSV_ridge(raw_data, utility_minimum_samples=10, gamma=None, is_upper=True):
    (X_dist_whitened, _, resi_dist), (X_star_whitened, _, resi_star), _ = preprocess_data(raw_data)
    sigma_2 = np.mean(resi_dist ** 2)

    DSV_list=[]
    for i in range(X_star_whitened.shape[0]):
        x_norm=sum(X_star_whitened[i]**2) # Assumed that a covariance matrix is the identity matrix.
        if is_upper is True:
            dsv=calculate_DSV_ridge_upper(x_norm=x_norm,
                                             m=X_star_whitened.shape[0], 
                                             input_dim=X_star_whitened.shape[1],
                                             utility_minimum_samples=utility_minimum_samples,
                                             gamma=gamma, 
                                             error_2=(resi_star[i]**2),
                                             sigma_2=sigma_2)
        else:
            dsv=calculate_DSV_ridge_lower(x_norm=x_norm,
                                             m=X_star_whitened.shape[0], 
                                             input_dim=X_star_whitened.shape[1],
                                             utility_minimum_samples=utility_minimum_samples, 
                                             gamma=gamma,
                                             error_2=(resi_star[i]**2),
                                             sigma_2=sigma_2)

        DSV_list.append(dsv)
    DSV_list = np.array(DSV_list)
    return DSV_list

def calculate_DSV_ridge_upper(x_norm=1.0,
                                 m=100,
                                 input_dim=5,
                                 utility_minimum_samples=10, 
                                 gamma=None,
                                 error_2=1, 
                                 sigma_2=1,
                                 tol=TOL,
                                 patience_limit=10):
    n_patience, nu_part = 0, 0
    small_c, big_c = 1, 1
    if gamma is None:
        gamma = 1/m

    for j in range(utility_minimum_samples-1,m):
        t_j = np.sqrt(np.log(j*np.sqrt(m))/small_c)
        delta_j = (big_c*np.sqrt(input_dim)+t_j)/np.sqrt(j)
        lambda_lower_j = 1/(j*((1+delta_j)**2)+gamma)
        lambda_upper_j = 1/(j*((1-delta_j)**2)+gamma)
        numer = 1+x_norm*lambda_lower_j
        denom = 1+x_norm*lambda_upper_j
        lambda_ratio = (numer/denom)**2
        core_part = x_norm*(lambda_upper_j**2)*((1+denom)*sigma_2-lambda_ratio*error_2)/(numer**2)
        nu_part_diff = core_part/m
        nu_part += nu_part_diff
        if np.abs((nu_part_diff)/(nu_part)) < tol:
            n_patience += 1
            if n_patience >= patience_limit:
                break    
    return nu_part


def calculate_DSV_ridge_lower(x_norm=1.0,
                                 m=100,
                                 input_dim=5,
                                 utility_minimum_samples=10, 
                                 gamma=None,
                                 error_2=1, 
                                 sigma_2=1,
                                 tol=TOL,
                                 patience_limit=10):
    n_patience, nu_part = 0, 0
    small_c, big_c = 1, 1
    if gamma is None:
        gamma = 1/m
        
    for j in range(utility_minimum_samples-1,m):
        t_j = np.sqrt(np.log(j*np.sqrt(m))/small_c)
        delta_j = (big_c*np.sqrt(input_dim)+t_j)/np.sqrt(j)
        lambda_lower_j = 1/(j*((1+delta_j)**2)+gamma)
        lambda_upper_j = 1/(j*((1-delta_j)**2)+gamma)
        numer = 1+x_norm*lambda_lower_j
        denom = 1+x_norm*lambda_upper_j
        lambda_ratio = (numer/denom)**2
        core_part = x_norm*(lambda_lower_j**2)*((1+numer)*sigma_2-error_2/lambda_ratio)/(denom**2)
        nu_part_diff = core_part/m
        nu_part += nu_part_diff
        if np.abs((nu_part_diff)/(nu_part)) < tol:
            n_patience += 1
            if n_patience >= patience_limit:
                break    
    return nu_part


