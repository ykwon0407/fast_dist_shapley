import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.datasets import load_boston, load_diabetes
from sklearn.datasets.samples_generator import make_classification
import pickle
from fastdist_utils import normalize_X
        
DATA_PATH = '/oak/stanford/groups/mrivas/users/yckwon/repos/DistributionalShapley/temp'
BINARY_PATH = '/home/users/yckwon/data/binary_dataset'
IMAGE_PATH='/home/users/yckwon/data/image_features'
REG_PATH='/home/users/yckwon/data/regression_dataset'

def load_regression_data(dataset='gaussian1'):
    m = 200 # number of data points to be valued.
    test_size = 1000
    heldout_size = 1000
    if dataset == 'gaussian1':
        print('-'*50)
        print('GAUSSIAN1')
        print('-'*50)
        # m=500 # number of data points to be valued.
        n=50000
        input_dim=10
        X_raw = np.random.normal(size=(n,input_dim))
        beta = np.random.normal(size=(input_dim,1))
        error_raw = np.random.normal(size=(n,1))
        Y_raw = X_raw.dot(beta) + error_raw
        data, target = X_raw, Y_raw
    elif dataset == 'gaussian2':
        print('-'*50)
        print('GAUSSIAN2')
        print('-'*50)
        # m=500 # number of data points to be valued.
        n=50000
        input_dim=30
        X_raw = np.random.normal(size=(n,input_dim))
        beta = np.random.normal(size=(input_dim,1))
        error_raw = np.random.normal(size=(n,1))
        Y_raw = X_raw.dot(beta) + error_raw
        data, target = X_raw, Y_raw
    elif dataset == 'abalone':
        print('-'*50)
        print('Abalone')
        print('-'*50)
        raw_data = pd.read_csv(REG_PATH+"/abalone.data", header=None)
        raw_data.dropna(inplace=True)
        data, target = pd.get_dummies(raw_data.iloc[:,:-1],drop_first=True).values, raw_data.iloc[:,-1].values
    elif dataset == 'whitewine':
        print('-'*50)
        print('whitewine')
        print('-'*50)
        raw_data = pd.read_csv(REG_PATH+"/winequality-white.csv",delimiter=";")
        raw_data.dropna(inplace=True)
        data, target = raw_data.values[:,:-1], raw_data.values[:,-1]
    elif dataset == 'redwine':
        print('-'*50)
        print('redwine')
        print('-'*50)
        test_size = 500
        heldout_size = 500
        
        raw_data = pd.read_csv(REG_PATH+"/winequality-red.csv",delimiter=";")
        raw_data.dropna(inplace=True)
        data, target = raw_data.values[:,:-1], raw_data.values[:,-1]
    elif dataset == 'boston':
        print('-'*50)
        print('boston')
        print('-'*50)
        test_size = 100
        heldout_size = 100
        
        boston = load_boston()
        data = np.array(boston.data, copy=True)
        target = np.array(boston.target, copy=True)
    elif dataset == 'diabetes':
        print('-'*50)
        print('diabetes')
        print('-'*50)
        test_size = 100
        heldout_size = 100
        
        diabetes = load_diabetes()
        data = np.array(diabetes.data, copy=True)
        target = np.array(diabetes.target, copy=True)
    elif dataset == 'airfoil':
        print('-'*50)
        print('airfoil')
        print('-'*50)
        test_size = 500
        heldout_size = 500
        
        raw_data = pd.read_csv(REG_PATH+"/airfoil_self_noise.dat", sep='\t', names=['X1','X2,','X3','X4','X5','Y'])
        data = raw_data.values[:,:-1]
        target = raw_data.values[:,-1]
    else:
        assert False, f"Check {dataset}"

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

def load_classification_data(dataset='gaussian1', specific_class=1):
    m = 200 # number of data points to be valued.
    test_size = 1000
    heldout_size = 1000
    if dataset == 'gaussian1':
        print('-'*50)
        print('GAUSSIAN-1')
        print('-'*50)
        n=50000
        mu=2
        input_dim=3
        target = np.array((np.random.uniform(size=n) > 0.5).astype(np.int32))
        data = np.random.normal(size=(n,input_dim))
        data[:,0] = data[:,0] + target * mu
    elif dataset == 'gaussian2':
        print('-'*50)
        print('GAUSSIAN-2')
        print('-'*50)
        n=50000
        input_dim=3
        data = np.random.normal(size=(n,input_dim))
        beta_true = np.array([2.0, 0.0, 0.0]).reshape(input_dim,1) # 2*np.random.normal(size=(input_dim,1))
        p_true = np.exp(data.dot(beta_true))/(1.+np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'covertype':
        print('-'*50)
        print('COVERTYPE')
        print('-'*50)
        # array([1, 2, 3, 4, 5, 6, 7], dtype=int32)
        data, target = fetch_covtype(data_home=DATA_PATH+'/scikit_data', download_if_missing=True, return_X_y=True)
        target = (target != specific_class) + 0.0
        target = target.astype(np.int32)
    elif dataset == 'skin_nonskin':
        print('-'*50)
        print('Skin_Nonsin')
        print('-'*50)
        data = np.load(BINARY_PATH+'/skin_nonskin.npy') # ((245057, 4), array([1., 2.], dtype=float32))
        data, target = data[:,1:], data[:,0]
        target = (target == 2) + 0.0
        target = target.astype(np.int32)
    elif dataset == 'diabetes_scale':
        print('-'*50)
        print('Diabetes_scale')
        print('-'*50)
        test_size = 100
        heldout_size = 100
        data = np.load(BINARY_PATH+'/diabetes_scale.npy') # ((759, 9), array([-1.,  1.], dtype=float32))
        data, target = data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
    elif dataset == 'australian_scale':
        print('-'*50)
        print('Australian_scale')
        print('-'*50)
        test_size = 100
        heldout_size = 100
        data = np.load(BINARY_PATH+'/australian_scale.npy') # ((449, 13), array([-1.,  1.], dtype=float32))
        data, target = data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
    elif dataset in ['cifar10', 'fashion', 'mnist']:
        print('-'*50)
        print(f'{dataset}')
        print('-'*50)
        tr_resnet = pickle.load(open(IMAGE_PATH+f"/{dataset}_tr_ext_pretrained_resnet.pkl", "rb"))
        te_resnet = pickle.load(open(IMAGE_PATH+f"/{dataset}_te_ext_pretrained_resnet.pkl", "rb"))

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
        pca = PCA(n_components=32)
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

def load_time_comparison_data(sample_size=200, dimension=10):
    print('-'*50,flush=True)
    print(f'GAUSSIAN: (m,p) : ({sample_size},{dimension})',flush=True)
    print('-'*50,flush=True)
    
    m = sample_size # number of data points to be valued.
    test_size = 1000
    heldout_size = 1000
    n = m + 50000
    input_dim=dimension
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

def load_time_comparison_clf_data(sample_size=200, dimension=10):
    print('-'*50,flush=True)
    print(f'GAUSSIAN: (m,p) : ({sample_size},{dimension})',flush=True)
    print('-'*50,flush=True)
    
    m = sample_size # number of data points to be valued.
    test_size = 1000
    heldout_size = 1000
    n = m + 50000
    input_dim=dimension
    mu=2

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



    