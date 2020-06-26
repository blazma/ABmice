
import numpy as np

def vcorrcoef(X,y):
    # correlation between the rows of the matrix X with dimensions (N x k) and a vector y of size (1 x k)
    # about 200 times faster than calculating correlations row by row
    Xm = np.reshape(np.nanmean(X,axis=1),(X.shape[0],1))
    i_nonzero = np.nonzero(Xm[:,0] != 0)[0]
    X_nz = X[i_nonzero,:]
    Xm_nz = Xm[i_nonzero,:]

    ym = np.nanmean(y)
    r_num = np.nansum((X_nz-Xm_nz)*(y-ym),axis=1)
    r_den = np.sqrt(np.nansum((X_nz-Xm_nz)**2,axis=1)*np.nansum((y-ym)**2))
    r = r_num/r_den
    return r

def nan_divide(a, b, where=True):
    'division function that returns np.nan where the division is not defined'
    x = np.zeros_like(a)
    x.fill(np.nan)
    x = np.divide(a, b, out=x, where=where)
    return x

def nan_add(a, b):
    'addition function that handles NANs by replacing them with zero - USE with CAUTION!'
    a[np.isnan(a)] = 0
    b[np.isnan(b)] = 0
    x = np.array(a + b)
    return x

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
