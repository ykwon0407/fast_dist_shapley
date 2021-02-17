import numpy as np
import pandas as pd
import pickle
from fast_dist_shap import normalize_X

def load_reg_data_for_point_addition(m=200,
                                     test_size=1000,
                                     heldout_size=1000,
                                     dataset='gaussian',
                                     reg_path='./datasets/reg_datasets'):
    '''
    This function loads regression datasets datasets for the point addition experiments.
    m: The number of data points to be valued.
    test_size: the number of data points for evaluation of the utility function.
    heldout_size: the number of data points for evaluation of performances in point addition experiments.
    reg_path: path to regression datasets

    You may need to download datasets first. Make sure to store in 'reg_path'.
    The datasets are avaiable at the following links.
    abalone: https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/
    airfoil: https://archive.ics.uci.edu/ml/machine-learning-databases/00291/
    whitewine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
    '''
    
    if dataset == 'gaussian':
        print('-'*50)
        print('GAUSSIAN-R')
        print('-'*50)
        n, input_dim = 50000, 10
        X_raw = np.random.normal(size=(n,input_dim))
        beta = np.random.normal(size=(input_dim,1))
        error_raw = np.random.normal(size=(n,1))
        Y_raw = X_raw.dot(beta) + error_raw
        data, target = X_raw, Y_raw
    elif dataset == 'abalone':
        print('-'*50)
        print('Abalone')
        print('-'*50)
        raw_data = pd.read_csv(reg_path+"/abalone.data", header=None)
        raw_data.dropna(inplace=True)
        data, target = pd.get_dummies(raw_data.iloc[:,:-1],drop_first=True).values, raw_data.iloc[:,-1].values
    elif dataset == 'whitewine':
        print('-'*50)
        print('whitewine')
        print('-'*50)
        raw_data = pd.read_csv(reg_path+"/winequality-white.csv",delimiter=";")
        raw_data.dropna(inplace=True)
        data, target = raw_data.values[:,:-1], raw_data.values[:,-1]
    elif dataset == 'airfoil':
        print('-'*50)
        print('airfoil')
        print('-'*50)
        test_size = 500
        heldout_size = 500
        
        raw_data = pd.read_csv(reg_path+"/airfoil_self_noise.dat", sep='\t', names=['X1','X2,','X3','X4','X5','Y'])
        data = raw_data.values[:,:-1]
        target = raw_data.values[:,-1]
    else:
        assert False, f"Check {dataset}."

    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs]

    X_train, y_train = data[:m], target[:m]
    X_dist, y_dist = data[m:-(test_size+heldout_size)], target[m:-(test_size+heldout_size)]
    X_test, y_test = data[-(test_size+heldout_size):], target[-(test_size+heldout_size):]

    print(f'number of samples: {len(X_dist)}')
    mean_dist, std_dist = np.mean(X_dist, axis=0), np.std(X_dist, axis=0)
    X_dist = normalize_X(X_dist, mean_dist, std_dist)
    X_train = normalize_X(X_train, mean_dist, std_dist)
    X_test = normalize_X(X_test, mean_dist, std_dist)

    return (X_dist, y_dist), (X_train, y_train), (X_test, y_test)

def load_non_reg_data_for_point_addition(m=200, 
                                         test_size=1000,
                                         heldout_size=1000, 
                                         dataset='gaussian', 
                                         specific_class=1,
                                         clf_path='./datasets/clf_datasets',
                                         img_path='./datasets/image_datasets'):
    '''
    This function loads classification (or density estimation) datasets for the point addition experiments.
    m: The number of data points to be valued.
    test_size: the number of data points for evaluation of the utility function.
    heldout_size: the number of data points for evaluation of performances in point addition experiments.
    clf_path: path to classification datasets.
    img_path: path to extracted features of image datasets.

    You may need to prepare datasets first. Please run 'prep_non_reg_data.py' first.
    As for the datasets 'cifar10', 'fashion' and 'mnist', 
    we extract features from trained weights using 'torchvision.models.resnet18(pretrained=True)'.
    '''
    if dataset == 'gaussian':
        print('-'*50)
        print('GAUSSIAN-C')
        print('-'*50)
        n, input_dim=50000, 3
        data = np.random.normal(size=(n,input_dim))
        beta_true = np.array([2.0, 0.0, 0.0]).reshape(input_dim,1)
        p_true = np.exp(data.dot(beta_true))/(1.+np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'skin_nonskin':
        print('-'*50)
        print('Skin_Nonsin')
        print('-'*50)
        # ((245057, 4), array([1., 2.], dtype=float32))
        data = np.load(clf_path+'/skin_nonskin.npy') 
        data, target = data[:,1:], data[:,0]
        target = (target == 2) + 0.0
        target = target.astype(np.int32)
    elif dataset == 'diabetes_scale':
        print('-'*50)
        print('Diabetes_scale')
        print('-'*50)
        test_size, heldout_size = 100, 100
        # ((759, 9), array([-1.,  1.], dtype=float32))
        data = np.load(clf_path+'/diabetes_scale.npy') 
        data, target = data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
    elif dataset == 'australian_scale':
        print('-'*50)
        print('Australian_scale')
        print('-'*50)
        test_size, heldout_size = 100, 100
        # ((449, 13), array([-1.,  1.], dtype=float32))
        data = np.load(clf_path+'/australian_scale.npy') 
        data, target = data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
    elif dataset in ['cifar10', 'fashion', 'mnist']:
        print('-'*50)
        print(f'{dataset}')
        print('-'*50)
        tr_resnet = pickle.load(open(img_path+f"/{dataset}_tr_ext_pretrained_resnet.pkl", "rb"))
        te_resnet = pickle.load(open(img_path+f"/{dataset}_te_ext_pretrained_resnet.pkl", "rb"))

        data, target = tr_resnet['data'], tr_resnet['targets']
        target = (target < 5) + 0.0
        target = target.astype(np.int32)
    else:
        assert False, f"Check {dataset}"

    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs]

    if dataset in ['cifar10', 'fashion', 'mnist']:
        X_train, y_train = data[:m], target[:m]
        X_dist, y_dist = data[m:], target[m:]

        data, target = te_resnet['data'], te_resnet['targets']
        target = (target < 5) + 0.0
        target = target.astype(np.int32)
        ind = np.random.permutation(len(data))
        data, target = data[ind], target[ind]
        X_test, y_test = data, target

        from sklearn.decomposition import PCA
        pca = PCA(n_components=32) # we pick first 32 PCs.
        pca.fit(X_dist)

        X_train = pca.transform(X_train)
        X_dist = pca.transform(X_dist)
        X_test = pca.transform(X_test)
    else:
        X_train, y_train = data[:m], target[:m]
        X_dist, y_dist = data[m:-(test_size+heldout_size)], target[m:-(test_size+heldout_size)]
        X_test, y_test = data[-(test_size+heldout_size):], target[-(test_size+heldout_size):]

    print(f'number of samples: {len(X_dist)}')
    mean_dist, std_dist = np.mean(X_dist, axis=0), np.std(X_dist, axis=0)
    X_dist = normalize_X(X_dist, mean_dist, std_dist)
    X_train = normalize_X(X_train, mean_dist, std_dist)
    X_test = normalize_X(X_test, mean_dist, std_dist)

    return (X_dist, y_dist), (X_train, y_train), (X_test, y_test)

def load_reg_data_for_time_comp(sample_size=200, dimension=10):
    '''
    This function loads regression datasets for time comparison experiments.
    (sample_size, dimension): (the number of data points to be valued, dimensionality of data)
    '''
    print('-'*50,flush=True)
    print(f'GAUSSIAN for regression: (m,p) : ({sample_size},{dimension})',flush=True)
    print('-'*50,flush=True)
    
    m = sample_size
    test_size, heldout_size = 1000, 1000
    n = m + 50000
    input_dim=dimension

    # Define data and targets
    X_raw = np.random.normal(size=(n,input_dim))
    beta = np.random.normal(size=(input_dim,1))
    error_raw = np.random.normal(size=(n,1))
    Y_raw = X_raw.dot(beta) + error_raw
    data, target = X_raw, Y_raw
    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs]

    X_train, y_train = data[:m], target[:m]
    X_dist, y_dist = data[m:-(test_size+heldout_size)], target[m:-(test_size+heldout_size)]
    X_test, y_test = data[-(test_size+heldout_size):], target[-(test_size+heldout_size):]

    print(f'number of samples: {len(X_dist)}')
    mean_dist, std_dist = np.mean(X_dist, axis=0), np.std(X_dist, axis=0)
    X_dist = normalize_X(X_dist, mean_dist, std_dist)
    X_train = normalize_X(X_train, mean_dist, std_dist)
    X_test = normalize_X(X_test, mean_dist, std_dist)

    return (X_dist, y_dist), (X_train, y_train), (X_test, y_test)

def load_clf_data_for_time_comp(sample_size=200, dimension=10):
    '''
    This function loads classification datasets for time comparison experiments.
    (sample_size, dimension): (the number of data points to be valued, dimensionality of data)
    '''
    print('-'*50,flush=True)
    print(f'GAUSSIAN for classification: (m,p) : ({sample_size},{dimension})',flush=True)
    print('-'*50,flush=True)
    
    m = sample_size
    test_size, heldout_size = 1000, 1000
    n = m + 50000
    input_dim, mu = dimension, 2

    # Define data and targets
    target = np.array((np.random.uniform(size=n) > 0.5).astype(np.int32))
    data = np.random.normal(size=(n,input_dim))
    data[:,0] = data[:,0] + target * mu
    idxs = np.random.permutation(len(data))
    data, target = data[idxs], target[idxs]

    X_train, y_train = data[:m], target[:m]
    X_dist, y_dist = data[m:-(test_size+heldout_size)], target[m:-(test_size+heldout_size)]
    X_test, y_test = data[-(test_size+heldout_size):], target[-(test_size+heldout_size):]

    print(f'number of samples: {len(X_dist)}')
    mean_dist, std_dist = np.mean(X_dist, axis=0), np.std(X_dist, axis=0)
    X_dist = normalize_X(X_dist, mean_dist, std_dist)
    X_train = normalize_X(X_train, mean_dist, std_dist)
    X_test = normalize_X(X_test, mean_dist, std_dist)

    return (X_dist, y_dist), (X_train, y_train), (X_test, y_test)    



    