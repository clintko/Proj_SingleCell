import numpy as np

head = lambda x, n = 6: x[:n]
tail = lambda x, n = 6: x[-n:]

def rand_row(dat, nfrac = 0.1, k = None):
    """randomly select rows with fraction (100 * nfrac)%; if k is specified choose k random rows"""
    # get number of rows selected
    nrow = dat.shape[0]
    if k:
        nrow_select = k
    else:
        nrow_select = np.ceil(nrow * nfrac)
        
    # randomly select rows using np.random.randint
    nrow_select = np.int(nrow_select) # convert to int to avoid error
    idx = np.random.randint(nrow, size = nrow_select)        
    return dat[idx, :]

def rand_col(dat, nfrac = 0.1, k = None):
    """randomly select columns with fraction (100 * nfrac)%; if k is specified choose k random columns"""
    tmp = dat.T
    tmp = rand_row(tmp, nfrac = nfrac, k = k)
    return tmp.T