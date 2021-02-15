import numpy as np
import pandas as pd
from time import time
from sklearn.linear_model import LinearRegression
from fastdist_utils import preprocess_data
from sklearn.neighbors import KernelDensity

'''
Linear regression 
'''

TOL = 0.001 # 0.005 in paper

def estimate_DSV_linear_core(x_norm, 
                             subset_size, 
                             input_dim, 
                             error_2, 
                             sigma_2, 
                             MC_limit=10000,
                             tol=0.005, # 0.01 in paper
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





"""

def old_calculate_expectations(x_norm, subset_size, p, n_rpt=10000):
    n_patience_cal = 0
    new_exp_1, new_exp_2 = 0., 0.
    
    for j in range(n_rpt):
        old_exp_1, old_exp_2 = new_exp_1, new_exp_2

        # Generate chi-square distribution with the `subset_size-p+1` degree of freedom
        chi_part=np.random.normal(size=(subset_size-p+1,1))
        chi_dist=chi_part.T.dot(chi_part)[0,0] 
        denom = (x_norm+chi_dist)

        new_exp_1 = (old_exp_1*j + (x_norm/(denom**2)))/(j+1)
        new_exp_2 = (old_exp_2*j + 1/denom)/(j+1)

        relative_diff = np.abs(new_exp_1+new_exp_2-old_exp_1-old_exp_2)/(old_exp_1+old_exp_2+1e-12) 

        if relative_diff < 0.01:
            n_patience_cal += 1
            if n_patience_cal >= 10:
                break
        
    return new_exp_1, new_exp_2

# Theorem 2
def calculate_dist_shapley_value(x_norm=1.0, m=100, p=5, q=10, error_2=1, sigma_2=1, n_rpt=10000, is_debug=False):
    initial = (sigma_2/m)*p - (sigma_2/m)*(p/(q-p-2))
    error_to_sigma_ratio = error_2/sigma_2
    
    part_1, part_2 = 0., 0.
    n_patience = 0
    for subset_size in range(q,m+1):
        expectation_1, expectation_2 = old_calculate_expectations(x_norm, subset_size, p, n_rpt)
        diff_part_1 = (1-error_to_sigma_ratio)*(subset_size-1.)*expectation_1/(subset_size-p)
        diff_part_2 = (subset_size-1.)*(1./(subset_size-p-1)-expectation_2)/(subset_size-p)

        part_1 += diff_part_1
        part_2 += diff_part_2
        if np.abs((diff_part_1+diff_part_2)/(part_1+part_2)) < 0.0001:
            n_patience += 1
            if n_patience >= 10:
                # print('Stop: ', subset_size-q) # around 300
                break
        
    if is_debug == True:
        return initial, (sigma_2/m)*part_1, (sigma_2/m)*part_2
    else:
        return initial + (sigma_2/m)*part_1 + (sigma_2/m)*part_2

    
def make_df_with_different_errors(m=100, p=5, q=10):
    x_norm_list = np.linspace(0,3*p, 300) # First 300 points
    value_list = []
    for x_norm in x_norm_list:
        value_zero=calculate_dist_shapley_value(x_norm=x_norm, m=m, p=p, q=q, error_2=0., sigma_2=1.)
        value_half=calculate_dist_shapley_value(x_norm=x_norm, m=m, p=p, q=q, error_2=0.5, sigma_2=1.)
        value_1=calculate_dist_shapley_value(x_norm=x_norm, m=m, p=p, q=q, error_2=1.0, sigma_2=1.)
        value_2=calculate_dist_shapley_value(x_norm=x_norm, m=m, p=p, q=q, error_2=2.0, sigma_2=1.)
        value_4=calculate_dist_shapley_value(x_norm=x_norm, m=m, p=p, q=q, error_2=4.0, sigma_2=1.)
        value_8=calculate_dist_shapley_value(x_norm=x_norm, m=m, p=p, q=q, error_2=8.0, sigma_2=1.)
        value_list.append([value_zero,value_half,value_1,value_2,value_4,value_8])    

    df = pd.DataFrame(np.concatenate((x_norm_list[:,np.newaxis],np.array(value_list)), axis=1))
    df.columns = ['x','Error=0','Error=0.5','Error=1','Error=2','Error=4','Error=8']
    return df

def deletion_test_with_order(descending_order_list, X_tr, y_tr, X_test, y_test, p=10, sigma_2=1, util_constant=2.):
    error_list = []
    m = X_tr.shape[0]
    baseline_constant = util_constant*sigma_2 # np.mean(y_tr ** 2) # Similar to sigma_2*(1+p)
    for ind, i in enumerate(descending_order_list):
        current_list = descending_order_list[ind:]
        X_batch, y_batch = X_tr[current_list], y_tr[current_list]
        model=LinearRegression(fit_intercept=False)
        model.fit(X_batch, y_batch)

        current_error = baseline_constant - np.mean((y_test - model.predict(X_test))**2)
        # current_error = sigma_2*(1+p) - np.mean((y_test - model.predict(X_test))**2)
        error_list.append(current_error)    
    return error_list
"""