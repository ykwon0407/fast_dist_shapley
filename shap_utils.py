import os, sys, warnings, inspect
import numpy as np
from scipy.stats import logistic
from scipy.stats import spearmanr
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.base import clone
from multiprocessing import dummy as multiprocessing
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate


def one_hotisze(X, impute=True, missing_key=-10000):
    X_oh = []
    for col in range(X.shape[-1]):
        column = X[:, col]
        vals = np.sort(list(set(column)))
        if impute and missing_key in vals:
            counts = np.zeros(len(vals))
            for i in range(len(vals)):
                counts[i] = np.sum(column == vals[i])
            column[column==missing_key] = vals[np.argmax(counts)]
        column_oh = np.zeros((len(column), len(vals)))
        for i, val in enumerate(np.sort(vals)):
            column_oh[column==val, i] = 1
        X_oh.append(column_oh)
    return np.concatenate(X_oh, -1)


def convergence_plots(marginals):
    plt.rcParams['figure.figsize'] = 15, 15
    for i, idx in enumerate(np.arange(min(25, marginals.shape[-1]))):
        plt.subplot(5,5,i+1)
        plt.plot(np.cumsum(marginals[:, idx])/np.arange(1, len(marginals)+1))    
        
    
def is_integer(array):
    return (np.equal(np.mod(array, 1), 0).mean()==1)


def is_fitted(model):
        """Checks if model object has any attributes ending with an underscore"""
        return 0 < len( [k for k,v in inspect.getmembers(model) if k.endswith('_') and not k.startswith('__')] )


def return_model(mode, **kwargs):
    if inspect.isclass(mode):
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = mode(**kwargs)
    elif mode=='logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        C = kwargs.get('C', 1.)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, C=C,
                                 max_iter=max_iter, random_state=666)
    elif mode=='Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif mode=='RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='SVC':
        kernel = kwargs.get('kernel', 'rbf')
        model = SVC(kernel=kernel, random_state=666)
    elif mode=='LinearSVC':
        model = LinearSVC(loss='hinge', random_state=666)
    elif mode=='GP':
        model = GaussianProcessClassifier(random_state=666)
    elif mode=='KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif mode=='NB':
        model = MultinomialNB()
    elif mode=='linear':
        model = LinearRegression()
    elif mode=='ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    else:
        raise ValueError("Invalid mode!")
    return model



def generate_features(latent, dependency):
    features = []
    n = latent.shape[0]
    exp = latent
    holder = latent
    for order in range(1, dependency+1):
        features.append(np.reshape(holder,[n,-1]))
        exp = np.expand_dims(exp,-1)
        holder = exp * np.expand_dims(holder,1)
    return np.concatenate(features,axis=-1)  


def label_generator(problem, X, param, difficulty=1, beta=None, important=None):
    if important is None or important > X.shape[-1]:
        important = X.shape[-1]
    dim_latent = sum([important**i for i in range(1, difficulty+1)])
    if beta is None:
        beta = np.random.normal(size=[1, dim_latent])
    important_dims = np.random.choice(X.shape[-1], important, replace=False)
    funct_init = lambda inp: np.sum(beta * generate_features(inp[:,important_dims], difficulty), -1)
    batch_size = max(100, min(len(X), 10000000//dim_latent))
    y_true = np.zeros(len(X))
    while True:
        try:
            for itr in range(int(np.ceil(len(X)/batch_size))):
                y_true[itr * batch_size: (itr+1) * batch_size] = funct_init(
                    X[itr * batch_size: (itr+1) * batch_size])
            break
        except MemoryError:
            batch_size = batch_size//2
    mean, std = np.mean(y_true), np.std(y_true)
    funct = lambda x: (np.sum(beta * generate_features(
        x[:, important_dims], difficulty), -1) - mean) / std
    y_true = (y_true - mean) / std
    if problem is 'classification':
        y_true = logistic.cdf(param * y_true)
        y = (np.random.random(X.shape[0]) < y_true).astype(int)
    elif problem is 'regression':
        y = y_true + param * np.random.normal(size=len(y_true))
    else:
        raise ValueError('Invalid problem specified!')
    return beta, y, y_true, funct


def one_iteration(clf, X, y, X_test, y_test, mean_score, tol=0.0, c=None, metric='accuracy'):
    """Runs one iteration of TMC-Shapley."""
    
    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")  
    if c is None:
        c = {i:np.array([i]) for i in range(len(X))}
    idxs, marginal_contribs = np.random.permutation(len(c.keys())), np.zeros(len(X))
    new_score = np.max(np.bincount(y)) * 1./len(y) if np.mean(y//1 == y/1)==1 else 0.
    start = 0
    if start:
        X_batch, y_batch =\
        np.concatenate([X[c[idx]] for idx in idxs[:start]]), np.concatenate([y[c[idx]] for idx in idxs[:start]])
    else:
        X_batch, y_batch = np.zeros((0,) +  tuple(X.shape[1:])), np.zeros(0).astype(int)
    for n, idx in enumerate(idxs[start:]):
        try:
            clf = clone(clf)
        except:
            clf.fit(np.zeros((0,) +  X.shape[1:]), y)
        old_score = new_score
        X_batch, y_batch = np.concatenate([X_batch, X[c[idx]]]), np.concatenate([y_batch, y[c[idx]]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_batch, y_batch)
                temp_score = score_func(clf, X_test, y_test)
                if temp_score>-1 and temp_score<1.: #Removing measningless r2 scores
                    new_score = temp_score
            except:
                continue
        marginal_contribs[c[idx]] = (new_score - old_score)/len(c[idx])
        if np.abs(new_score - mean_score)/mean_score < tol:
            break
    return marginal_contribs, idxs


def marginals(clf, X, y, X_test, y_test, c=None, tol=0., trials=3000, mean_score=None, metric='accuracy'):
    
    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:,1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")  
    if mean_score is None:
        accs = []
        for _ in range(100):
            bag_idxs = np.random.choice(len(y_test), len(y_test))
            accs.append(score_func(clf, X_test[bag_idxs], y_test[bag_idxs]))
        mean_score = np.mean(accs)
    marginals, idxs = [], []
    for trial in range(trials):
        if 10*(trial+1)/trials % 1 == 0:
            print('{} out of {}'.format(trial + 1, trials))
        marginal, idx = one_iteration(clf, X, y, X_test, y_test, mean_score, tol=tol, c=c, metric=metric)
        marginals.append(marginal)
        idxs.append(idx)
    return np.array(marginals), np.array(idxs)


def shapley(mode, X, y, X_test, y_test, stop=None, tol=0., trials=3000, **kwargs):
    
    try:
        vals = np.zeros(len(X))
        example_idxs = np.random.choice(len(X), min(25, len(X)), replace=False)
        example_marginals = np.zeros((trials, len(example_idxs)))
        for i in range(trials):
            print(i)
            output = one_pass(mode, X, y, X_test, y_test, tol=tol, stop=stop, **kwargs)
            example_marginals[i] = output[0][example_idxs]
            vals = vals/(i+1) + output[0]/(i+1)
        return vals, example_marginals
    except KeyboardInterrupt:
        print('Interrupted!')
        return vals, example_marginals

    
def early_stopping(marginals, idxs, stopping):
    
    stopped_marginals = np.zeros_like(marginals)
    for i in range(len(marginals)):
        stopped_marginals[i][idxs[i][:stopping]] = marginals[i][idxs[i][:stopping]]
    return np.mean(stopped_marginals, 0)


def error(mem):
    
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)


def my_accuracy_score(clf, X, y):
    
    probs = clf.predict_proba(X)
    predictions = np.argmax(probs, -1)
    return np.mean(np.equal(predictions, y))


def my_f1_score(clf, X, y):
    
    predictions = clf.predict(x)
    if len(set(y)) == 2:
        return f1_score(y, predictions)
    return f1_score(y, predictions, average='macro')


def my_auc_score(clf, X, y):
    
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    return roc_auc_score(y, true_probs)


def my_xe_score(clf, X, y):
    
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    true_log_probs = np.log(np.clip(true_probs, 1e-12, None))
    return np.mean(true_log_probs)


def portion_performance(dshap, order, points, X_new, y_new, X_init, y_init, X_test, y_test):
    # Data are added
    dshap.model.fit(X_init, y_init)
    val_init = dshap.value(dshap.model, dshap.metric, X=X_test, y=y_test)
    vals = [val_init]
    for point in points:
        if not point:
            continue
        dshap.model.fit(np.concatenate([X_init, X_new[order[:point]]]),
                       np.concatenate([y_init, y_new[order[:point]]]))
        vals.append(dshap.value(dshap.model, dshap.metric, X=X_test, y=y_test))
    return np.array(vals)  

def portion_performance_density(dshap, order, points, X_new, X_init, X_test):
    # Data are added
    random_score = dshap.random_score
    dshap.model.fit(X_init)
    val_init = dshap.value(dshap.model, X=X_test)
    vals = [val_init - random_score]
    for point in points:
        if not point:
            continue
        dshap.model.fit(np.concatenate([X_init, X_new[order[:point]]]))
        vals.append(dshap.value(dshap.model, X=X_test)-random_score)
    return np.array(vals)  


def find_best_regressor(X, y, vals, cv=10, verbose=False):
    
    def return_model(model_family):
        
        if model_family == 'Ridge':
            params = 10 ** np.arange(0, -6, 1).astype(float)
            model = lambda param: Ridge(alpha=param)
        if model_family == 'Lasso':
            params = 10 ** np.arange(0, -6, 1).astype(float)
            model = lambda param: Lasso(alpha=param)
        if model_family == 'RF':
            params = [5, 10, 25, 50, 100]
            model = lambda param: RandomForestRegressor(n_estimators=param)
        if model_family == 'KNN':
            params = np.arange(1, 9, 2)
            model = lambda param: KNeighborsRegressor(n_neighbors=param)
        return model, params
    
    regs = {}
    for label in np.sort(list(set(y))):
        best_score = -100
        label_idxs = np.where(y == label)[0]
        if len(label_idxs) == 1:
            model, params = return_model('RF')
            best_reg = model(10)
            best_reg.fit(X[label_idxs], vals[label_idxs])
            regs[label] = best_reg
            continue
        for model_family in ['RF', 'Lasso', 'Ridge', 'KNN']:
            model, params = return_model(model_family)
            for param in params:
                if verbose:
                    print(model_family, param)
                reg = model(param)
                cv_scores = cross_validate(
                    reg,
                    X[label_idxs],
                    vals[label_idxs],
                    cv=min(len(label_idxs), cv))
                if np.mean(cv_scores['test_score']) > best_score:
                    best_score = np.mean(cv_scores['test_score'])
                    best_reg = reg
        print(label, best_reg, best_score)
        best_reg.fit(X[label_idxs], vals[label_idxs])
        regs[label] = best_reg
    return regs


def predict_vals(X, y, regs):
    
    predicted_vals = np.zeros(len(X))
    for label in set(y):
        label_idxs = np.where(y == label)[0]
        predicted_vals[label_idxs] = regs[label].predict(X[label_idxs])
    return predicted_vals
    
def s_regress(model, vals, alpha, init):
    
    predicted_vals = np.zeros(truncation)
    t = int(truncation ** (1./(1 + alpha)))
    t_idxs = (np.arange(t) ** (1+alpha)).astype(int)
    t_idxs = np.sort(np.array(list(set(t_idxs))))
    t_idxs = t_idxs[t_idxs>=init]
    x = t_idxs
    y = vals[t_idxs]
    model.fit(x, y)
    predicted_vals = model.predict(np.arange(truncation))
    predicted_vals[:init] = vals[:init]
    predicted_vals[t_idxs] = vals[t_idxs]
    return predicted_vals

def interpolator(model):
    
    if 'spline_' in model:
        return Spline(model[7:])
    if model == 'lin':
        return LinInt()
    if 'poly_' in model:
        return Poly(int(model[5:]))
    if model == 'NN':
        return NN(activation='logistic')
    raise ValueError('Invalid Model')

    
def compute_eff(alpha, truncation):
    
    t = int(truncation ** (1./(1 + alpha)))
    t_idxs = (np.arange(t) ** (1+alpha)).astype(int)
    t_idxs = np.sort(np.array(list(set(t_idxs))))
    t_idxs = t_idxs[t_idxs>=init]
    return (np.sum(np.arange(init)) + np.sum(t_idxs)) / np.sum(np.arange(truncation)) 


def reverse_compute_eff(x, truncation):
    
    a1 = 0.
    a2 = 10.
    while True:
        if compute_eff((a1+a2)/2, truncation) < x:
            a1, a2 = a1, (a1+a2)/2
        else:
            a1, a2 = (a1+a2)/2, a2
        if a2 - a1 < 1e-4:
            break
    return (a1+a2)/2

def performance_plots(npoints, points, perf):
    
    plt.rcParams['font.size'] = 15
    fig = plt.figure(figsize = (16, 8))
    markers = ['-', ':', '-.', '--']
    colors = ['b', 'r', 'g', 'orange']
    default_legends = ['Dist-Shapley', 'Random', 'LOO', 'TMC-Shapley']
    
    plt.subplot(1, 2, 1)
    pos_keys = ['pos_dist', 'rnd', 'pos_loo', 'pos_tmc']
    legends = []
    for i, (key, legend) in enumerate(zip(pos_keys, default_legends)):
        if key not in perf:
            continue
        plt.plot(points / npoints * 100, 100 * np.array(perf[key]), markers[i], color=colors[i], lw=8)
        legends.append(legend)
    plt.legend(legends, fontsize=25)
    res = (points[-1] - points[0]) / (len(points) - 1)
    plt.xticks(100 * np.linspace(points[0], points[-1] + res, 6) / npoints)
    plt.xlabel('Fraction of points removed (%)', fontsize=25)
    min_p = np.min([np.min(perf[k]) for k in perf if k in pos_keys])
    max_p = np.max([np.max(perf[k]) for k in perf if k in pos_keys])
    p_res = 0.01
    for p in [0.2, 0.1, 0.05, 0.03, 0.02, 0.01]:
        num_p = np.ceil(max_p / p) - np.floor(min_p / p)
        if num_p >= 4 and num_p <= 8:
            p_res = p
            break
    plt.yticks(100 * np.arange(np.floor(min_p/p_res) * p_res, np.ceil(max_p/p_res) * p_res + 0.01, p_res))
    plt.ylabel('Performance (%)', fontsize=25)
    
    plt.subplot(1, 2, 2)
    legends = []
    neg_keys = ['neg_dist', 'rnd', 'neg_loo', 'neg_tmc']
    for i, (key, legend) in enumerate(zip(neg_keys, default_legends)):
        if key not in perf:
            continue
        plt.plot(points / npoints * 100, 100 * np.array(perf[key]), markers[i], color=colors[i], lw=8)
        legends.append(legend)
    plt.legend(legends, fontsize=25)
    res = (points[-1] - points[0]) / (len(points) - 1)
    plt.xticks(100 * np.linspace(points[0], points[-1] + res, 6) / npoints)
    plt.xlabel('Fraction of points removed (%)', fontsize=25)
    min_p = np.min([np.min(perf[k]) for k in perf if k in neg_keys])
    max_p = np.max([np.max(perf[k]) for k in perf if k in neg_keys])
    p_res = 0.01
    for p in [0.2, 0.1, 0.05, 0.03, 0.02, 0.01]:
        num_p = np.ceil(max_p / p) - np.floor(min_p / p)
        if num_p >= 4 and num_p <= 8:
            p_res = p
            break
    plt.yticks(100 * np.arange(np.floor(min_p/p_res) * p_res, np.ceil(max_p/p_res) * p_res + 0.01, p_res))
    plt.ylabel('Performance (%)', fontsize=25)
