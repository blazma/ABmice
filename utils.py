
import numpy as np

def vcorrcoef(X,y, zero_var_out=0):
    # correlation between the rows of the matrix X with dimensions (N x k) and a vector y of size (1 x k)
    # zero_var_out: is the output where the variance is 0
    # can be either 0 or np.nan
    # about 200 times faster than calculating correlations row by row
    Xm = np.reshape(np.nanmean(X,axis=1),(X.shape[0],1))
    vec_nonzero = np.sum((X - Xm)**2, axis=1) != 0 # we select the rows with nonzera VARIANCE...

    ym = np.nanmean(y)
    r_num = np.nansum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.nansum((X-Xm)**2,axis=1)*np.nansum((y-ym)**2))
    out_vec = np.zeros_like(r_num)
    out_vec[:] = zero_var_out
    r = np.divide(r_num, r_den, out=out_vec, where=vec_nonzero)
    return r


def Mcorrcoef(X,Y, zero_var_out=0):
    # correlation between the rows of two matrices X and Y with dimensions (N x k)
    # zero_var_out: is the output where the variance is 0
    # can be either 0 or np.nan

    Xm = np.reshape(np.nanmean(X,axis=1),(X.shape[0],1))
    Ym = np.reshape(np.nanmean(Y,axis=1),(Y.shape[0],1))
    vec_nonzero = (np.sum((X - Xm)**2, axis=1) != 0) & (np.sum((Y - Ym)**2, axis=1) != 0) # we select the rows with nonzera VARIANCE...

    r_num = np.nansum((X-Xm)*(Y-Ym),axis=1)
    r_den = np.sqrt(np.nansum((X-Xm)**2,axis=1)*np.nansum((Y-Ym)**2,axis=1))
    out_vec = np.zeros_like(r_num)
    out_vec[:] = zero_var_out
    r = np.divide(r_num, r_den, out=out_vec, where=vec_nonzero)
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


def test_vcorrcoeff():
    X = np.arange(24).reshape(6,4)
    y = np.array([3,4,1,2]).reshape(1,4)
    X[1,:] = 0
    X[2,:] = 13
    X[3,:] = y
    X[4,:] = 1-y
    X[5,1] = 13
    r = vcorrcoef(X, y)

    if ((np.corrcoef(X[0,:], y)[1,0] - r[0]) > 1e-10):
        print('Error 0')
        return False

    if (0 != r[1]):
        print('Error 1')
        return False

    if (0 != r[2]):
        print('Error 2')
        return False

    if ((np.corrcoef(X[3,:], y)[1,0] - r[3]) > 1e-10):
        print('Error 3')
        return False

    if ((np.corrcoef(X[4,:], y)[1,0] - r[4]) > 1e-10):
        print('Error 4')
        return False

    if ((np.corrcoef(X[5,:], y)[1,0] - r[5]) > 1e-10):
        print('Error 5')
        return False

    return True


def test_Mcorrcoeff():
    X = np.arange(24).reshape(6,4)
    Y = np.arange(24).reshape(6,4)
    X[:,2] = 5
    X[1,:] = 0
    X[2,:] = 13
    X[3,:] = Y[3,:]
    X[4,:] = 1-Y[4,:]
    r = Mcorrcoef(X, Y)

    if ((np.corrcoef(X[0,:], Y[0,:])[1,0] - r[0]) > 1e-10):
        print('Error 0')
        return False

    if (0 != r[1]):
        print('Error 1')
        return False

    if (0 != r[2]):
        print('Error 2')
        return False

    if ((np.corrcoef(X[3,:], Y[3,:])[1,0] - r[3]) > 1e-10):
        print('Error 3')
        return False

    if ((np.corrcoef(X[4,:], Y[4,:])[1,0] - r[4]) > 1e-10):
        print('Error 4')
        return False

    if ((np.corrcoef(X[5,:], Y[5,:])[1,0] - r[5]) > 1e-10):
        print('Error 5')
        return False

    return True
    
test_vcorrcoeff()
test_Mcorrcoeff()
