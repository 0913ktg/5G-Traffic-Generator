from scipy.stats import entropy
import numpy as np
import torch

'''
Jensen–Shannon divergence

관련 정보
- https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

'''

def jsd(target, pred):

    target = np.array(sorted(target))
    pred = np.array(sorted(pred))

    # target    
    mu, sigma = target.mean(), target.std()
    wb1 = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(target-mu)**2 / (2 * sigma**2))
    
    # predict
    mu, sigma = pred.mean(), pred.std()
    wb2 = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(pred-mu)**2 / (2 * sigma**2))
        
    p = np.asarray(wb1)
    q = np.asarray(wb2)
    
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


'''
maximum mean discrepancy square with Gaussian RBF kernel

관련 정보
- https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py

'''

def rbf_mmd2(target, pred, sigma=1, biased=True):        
    gamma = 1 / (2 * sigma**2)
        
    pred = pred.reshape(-1,1)
    target = target.reshape(-1,1)
    
    X = torch.from_numpy(pred)
    Y = torch.from_numpy(target)    
    X = X.type(torch.float64)
    Y = Y.type(torch.float64)

    XX = torch.matmul(X, X.T)
    XY = torch.matmul(X, Y.T)
    YY = torch.matmul(Y, Y.T)

    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)

    K_XY = torch.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = torch.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = torch.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return float(mmd2)


if __name__ == "__main__":
    
    target = np.random.randint(1,11,size = 20)
    pred = np.random.randint(1,11,size = 20)
    
    print(f'jsd score : {jsd(target, pred)}')
    print(f'mmd_square score : {rbf_mmd2(target, pred)}')