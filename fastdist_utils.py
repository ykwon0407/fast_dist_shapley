import statsmodels.api as sm
import numpy as np
from scipy.stats import spearmanr

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


def print_rank_correlation(vals_tmc, vals_dist, vals_fastdist):
    print('-'*30)
    print('Rank correlation vs random')
    print('-'*30)

    val_list = [vals_tmc, vals_dist, vals_fastdist]
    name_list = ['TMC','D-Shapley','FastDist']

    for i in range(3): 
        corr = spearmanr(np.random.normal(size=len(vals_fastdist)), val_list[i])[0]
        print(f'Rank correlation: {name_list[i]} vs random values = {corr:.3f}')

    print('-'*30)
    print('Rank correlation')
    print('-'*30)

    for i in range(3):
        for j in range(3):
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





