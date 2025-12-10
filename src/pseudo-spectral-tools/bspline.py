import numpy as np
from pathos.multiprocessing import ProcessPool
import pathos as pa
# just keep functions in this script. all other codes will call this script.

# For value checking, make a separate function to check inputs 

def V(k,i,knotArr,x):
    '''
    Computes the coffecients in the B-spline recursion relation

    Parameters
    ----------
    k : integer
        order of the b-spline.
    i : integer
        label of the i-th b-spline.
    knotArr : ndarry
        array containing knot positions.
    x : ndarray
        array containing collocation points.

    Returns
    -------
    V : ndarray
        array containing V coefficients for position array x.

    '''
    # returns V(i,x) as a (maxi,len(x)) array. Each col corresponds to  a V_i(x)
    #V = np.zeros((len(x)))
    V = (x- knotArr[i])/(knotArr[i + k] - knotArr[i]) 
    if np.isinf(V).any() or np.isnan(V).any():
       # print(f'infinity in B-spline at k = {k}. Setting it to zero.')
        V = np.zeros(len(x))
    return V
def Bspl(k,i,knotArr,x):
    '''

    Parameters
    ----------
    k : integer
        order of the b-spline.
    i : integer
        integer labeling the spline associated with knot i.
    knotArr : ndarray
        array of knot locations.
    x : ndarray
        collocation points.

    Raises
    ------
    ValueError
        raises a value error if the order of the b-spline and number of knots are 
        incompatible.

    Returns
    -------
    B : ndarray
        vector containing B_{i}(x_{j}) for fixed i.

    '''
    knotArr = np.around(knotArr,5)
    x = np.around(x,5)
    if i > len(knotArr)-1:
        raise ValueError('Bspline can only be evaluated for max_i = len(knots) - k -1')
    if k <= 0:
        raise(ValueError("k must be >= 1"))
    if k == 1:
        x_inds = np.where((x < knotArr[i+1]) & (x>= knotArr[i]))
        B = np.zeros(len(x))
        B[x_inds] = 1.0
        return B
    else:
        B = V(k-1,i,knotArr,x)*Bspl(k-1,i,knotArr,x) + (1.0 - V(k-1,i+1,knotArr,x))*Bspl(k-1,i+1,knotArr,x)
        return B
def getBspline(k,knotArr,x,bndry,multiprocessing = False):
    '''
    Constructs the B-spline matrix B_{i}(x_{j})
    Parameters
    ----------
    k : integer
        order of the b-spline.
    knotArr : ndarray
        array of knot locations.
    x : ndarray 
        array of collocation points.
    multiprocessing : bool, optional
        use multiprocessing to construct the B-matrix. It distrubutes evaluation of 
        B_{i}(x_{j}) over i. WARNING: This option currently has a bug that causes
        a race condition
        The default is False.

    Raises
    ------
    ValueError
        checks if spline order is compatible with number of knots.

    Returns
    -------
    ndarry
        returns B_{i}(x_{j}) matrix. Columns are labeled by i and rows by j.

    '''
    if k >= len(knotArr):
        raise ValueError("Spline order must be < len(knotArr)")
    max_i = len(knotArr) - k 
    #dynamic processing? use multiprocess
    if multiprocessing == True:  
        def wrapper(i):
            return Bspl(k,i,knotArr,x)
        n_nodes = pa.helpers.cpu_count() 
        with ProcessPool(nodes=n_nodes) as pool:
            BspArr = pool.map(wrapper, np.arange(0,max_i))
        BspArr = np.array(BspArr)
        if bndry == 'dirichlet':
            return BspArr.T
        elif bndry == 'periodic':
            BspArr = BspArr.T
            #for i in range(0,k-1):   
            #    num = (k - 1) - i  
            #    BspArr[:,i] +=  BspArr[:,-num] 
            num = range(0,k-1)
            num_neg = range(-(k-1),0)
            BspArr[:,num] += BspArr[:,num_neg]
            BspArr = BspArr[:,:len(x)]
            return BspArr
    else:
        BspArr = np.array([Bspl(k,i,knotArr,x) for i in range(max_i)])
        if bndry == 'dirichlet':
            return BspArr.T
        elif bndry == 'periodic':
            BspArr = BspArr.T
            #for i in range(0,k-1):   
            #    num = (k - 1) - i  
            #    print(i,-num)
            #    BspArr[:,i] +=  BspArr[:,-num] 
            
            num = range(0,k-1)
            num_neg = range(-(k-1),0)
            BspArr[:,num] += BspArr[:,num_neg]
            BspArr = BspArr[:,:len(x)]
            return BspArr
def extendKnots(splOrder,knotArr,method='extension'):
    # This adds the appropriate amount of extra knots to the boundary based on 
    # what the spline order is
    # splOrder is the order of spline used
    # knotArr is the array containing the x value of all the knots in the 
    # PHYSICAL region.
    N = len(knotArr)
    if N <= splOrder:
        raise ValueError('Spline order k needs to be greater than the number of knots')
    if splOrder == 1:
        return knotArr
    if method=='extension':
        #This method extends bsplines outside of the physical domain by mirroring 
        #first splOrder points outside the physical boundary and last N + splOrder
        # outside the physical boundary
        shift_i = knotArr[0] - knotArr[splOrder - 1]
        shift_f = knotArr[N-1] - knotArr[N -1 - (splOrder-1)]
        if shift_i > 0:
            extraKnots_i = knotArr[:splOrder-1] - shift_i
        else: 
            extraKnots_i = knotArr[:splOrder-1] + shift_i
        if shift_f > 0:
            extraKnots_f = knotArr[N - 1 -(splOrder-2):] + shift_f
        else:
            extraKnots_f = knotArr[N - 1 - (splOrder-2):] - shift_f
        knotArr = np.concatenate((extraKnots_i, knotArr,extraKnots_f))
    elif method=='bndry':
        # This method just sticks multiple knots at the physical boundary location
        extraKnots_i = np.full(splOrder-1, knotArr[0])
        extraKnots_f = np.full(splOrder-1, knotArr[-1])
        knotArr = np.concatenate((extraKnots_i, knotArr,extraKnots_f))
    return knotArr
def _makeC(k,i,knotArr,x):
    # computes d^k /dx^k B^k_i(x)
    if k == 1:
        x_inds = np.where((x < knotArr[i+1]) & (x>= knotArr[i]))
        C = np.zeros(len(x))
        C[x_inds] = 1.0
        return C
    C = (k-1)*(_makeC(k-1,i,knotArr,x)/(knotArr[k -1 + i] - knotArr[i]) \
           - _makeC(k-1,i+1,knotArr,x)/(knotArr[k -1 + i +1] - knotArr[i+1]))
    if np.isinf(C).any() or np.isnan(C).any():
       # print(f'infinite detected in B-spline at k = {k}. Setting it to zero.')
        C = np.zeros(len(x))
    return C
def getC(k,knotArr,x):
    if len(knotArr) <= k:
        raise ValueError('Spline order k needs to be greater than the number of knots')
    max_i = len(knotArr) - k 
    CArr = np.array([_makeC(k,i,knotArr,x) for i in range(max_i)])
    return CArr 
def _makedpdxp(p,k,i,knotArr,x):
    # computes d^{p}/dx^{p} B^{k}_{i}(x) for a given i 
    if p > k - 1:
        return np.zeros(len(x))
    if k == p + 1 :
        dpBdxp = _makeC(k,i,knotArr,x)
        return dpBdxp
    dpBdxp = ((k-1)/(k - 1 -p))*(V(k-1,i,knotArr,x)*_makedpdxp(p,k-1,i,knotArr,x) + (1.0 - V(k-1,i+1,knotArr,x))*_makedpdxp(p,k-1,i+1,knotArr,x))
    return dpBdxp 
def getDp(p,k,knotArr,x,bndry='dirichlet',multiprocessing = False):
    # computes d^{p}/dx^{p} B_{i}^{k}(x) for all i 
    max_i = len(knotArr) - k
    if multiprocessing == True:
        def wrapper(i):
            return _makedpdxp(p,k,i,knotArr,x) 
        n_nodes = pa.helpers.cpu_count() 
        with ProcessPool(nodes=n_nodes) as pool:
            dBdxArr = pool.map(wrapper, np.arange(0,max_i))
        dBdxArr = np.array(dBdxArr)
        if bndry =='dirichlet':
            return dBdxArr.T
        elif bndry == 'periodic':
            dBdxArr = dBdxArr.T
            #for i in range(0,k-1):   
            #    num = (k - 1) - i  
            #    dBdxArr[:,i] +=  dBdxArr[:,-num] 
            num = range(0,k-1)
            num_neg = range(-(k-1),0)
            dBdxArr[:,num] += dBdxArr[:,num_neg]
            dBdxArr = dBdxArr[:,:len(x)]
            return dBdxArr
        else: 
            raise(ValueError('bndry = dirichlet or periodic'))
    else:
        if bndry=='periodic':
            dBdxArr = np.array([_makedpdxp(p,k,i,knotArr,x) for i in range(max_i)])
            dBdxArr = dBdxArr.T
            num = range(0,k-1)
            num_neg = range(-(k-1),0)
            dBdxArr[:,num] += dBdxArr[:,num_neg]
            dBdxArr = dBdxArr[:,:len(x)]
            return dBdxArr
        elif bndry =='dirichlet':
            dBdxArr = np.array([_makedpdxp(p,k,i,knotArr,x) for i in range(max_i)])
            return dBdxArr.T
def BMatrix(k,knotArr,x,bndry,multiprocessing = False):
    # generates the matrix B^{k}_{i \alpha} = B^{k}_{i}(x_{\alpha})
    # This is just the array of values of B_{i} evaluated at the collocation points
    BsplArr = getBspline(k,knotArr,x,bndry=bndry,multiprocessing = multiprocessing)
    return BsplArr

def getKnots(xi,xf,N_knots,splOrder,knotFunc,knotFunc_params,extensionMethod='extension'):
    knotArr = knotFunc(xi,xf,N_knots,knotFunc_params)
    knotArr = extendKnots(splOrder,knotArr,method=extensionMethod)
    delta_x_knots = np.array([abs(knotArr[i+1]- knotArr[i]) for i in range(len(knotArr)-1)])
    return knotArr,delta_x_knots

def getCPnts(splOrder,knotArr,bndryPnts):
    # This will assume an odd ordered spline and will put collocation points
    # at the mid point of each knot
    if splOrder % 2 == 0:
        raise(ValueError('Spline order can only be odd.'))
    cPnts = []
    #bndryPnts = list(bndryPnts)
    bndryArr = []
    # Every odd order higher than splOrder 3, we need to add 2 additional collocation
    # points
    for i in np.arange(splOrder,len(knotArr)-splOrder +1):
        cPnts.append((knotArr[i-1] + knotArr[i])/2.0)
    if splOrder > 3:
        diff = int(.5*(splOrder - 3))
    else:
        diff = 0
    startInd = splOrder  - diff
    endInd = len(knotArr)-splOrder + 1 + diff
 
    for i in np.arange(startInd,startInd + diff):
        bndryArr.append((knotArr[i-1] + knotArr[i])/2.0)
    bndryArr.append(bndryPnts[0])
    bndryArr.append(bndryPnts[1])
    for i in np.arange(endInd - diff,endInd):
        bndryArr.append((knotArr[i-1] + knotArr[i])/2.0)    
    return np.array(cPnts),np.array(bndryArr)

def betaMatrix(splOrder,knotArr,BArr,K,bndry_pnts,bndry,multiprocessing = False):
    #constructs the matrix that will contain boundary conditions
    #beta_{ri} = \sum_{p \geq 0} K_{rp} \partial_{p} B^{k}_{i}(x) for x on boundary
    # K determines the type of BC. For example for two dirichelt BCs the size is (2,1)
    # K = {{1},{1}} = Dirchlet both sides
    beta = np.zeros((bndry_pnts.shape[0],BArr.shape[1]))

    for r in range(beta.shape[0]):
        p = np.where(K[r] > 0)[0][0]
        dBdx = getDp(p, splOrder, knotArr, np.array([bndry_pnts[r]]),bndry=bndry,multiprocessing =multiprocessing) 
        beta[r] = dBdx    
    B_tilde = np.concatenate((BArr, beta))
    return B_tilde,beta

def getKmatrix(splOrder,bndry):
    if bndry =='dirichlet':
        K = np.full((splOrder - 1,1),1)
    elif bndry == 'neumann':
        K = np.zeros((splOrder - 1,2))
        # the first two rows are always the physical boundary 
        for i in np.arange(0,splOrder - 1):
            K[i][0] = 1 
    elif bndry == 'robin':
        K = np.zeros((splOrder - 1,2))
        # the first two rows are always the physical boundary 
        K[0][0] = 1
        K[0][1] = 1
        K[1][0] = 1
        K[1][1] = 1
        # enforce dirichlet BCs on the points outside the physical region
        for i in np.arange(2,splOrder - 1):
            K[i][0] = 1 
    elif bndry == 'mixed':
        raise(ValueError('mixed BCs not implemented yet'))
        #K = np.zeros((splOrder - 1,2))    
    
    else:
        raise(ValueError('Only available BCs are dirichlet,neumann, and mixed'))
    return K
def getCtilde(B_tilde,splOrder,bndry,bndryVals=None,returnInv=False):
    B_tilde_inv = np.linalg.inv(B_tilde)
    if bndry=='dirichlet':
        if bndryVals[0] != 0.0 or bndryVals[-1] != 0.0 :
            C_tilde = B_tilde_inv 
        else:
            C_tilde = B_tilde_inv[:,:(len(B_tilde_inv)-splOrder+1)]
    elif bndry=='periodic':
        C_tilde = B_tilde_inv
    else: 
        raise(ValueError('bndry= periodic or dirichlet'))
    if returnInv == True:
        return C_tilde,B_tilde_inv
    else:
        return C_tilde
def getDMatrix(order,splOrder,knotArr,xArr,C_tilde,bndry,multiprocessing=False):
    if isinstance(order, int) == False:
        raise(ValueError('order of the derivative must be an integer >= 1'))
    dnBdxnArr = getDp(order, splOrder, knotArr, xArr,bndry=bndry, multiprocessing = multiprocessing)
    DMatrix = np.matmul(dnBdxnArr,C_tilde)
    return DMatrix#,dnBdxnArr

def getWeights(knotArr,C_tilde,splOrder,bndry):
    if bndry == 'dirichlet':
        hArr = np.zeros(len(knotArr)-splOrder)
        for i in range(0,len(knotArr)-splOrder):
            hArr[i] = (knotArr[i+splOrder] - knotArr[i])/splOrder
        wArr = np.matmul(C_tilde.T,hArr)
        
    elif bndry == 'periodic':
        hArr = np.zeros(len(knotArr)- 2*splOrder +1)
        for i in range(0,len(knotArr)-2*splOrder+1):
            hArr[i] = (knotArr[i+splOrder] - knotArr[i])/splOrder
        wArr = np.matmul(C_tilde,hArr)
    return wArr

def integrate(funcArr,wArr):
    #NOTE WE CAN ONLY INTEGRATE FUNCTIONS THAT VANISH AT INFINITY
    # if we want to be able to integrate over an arbitrary interval [a,b], 
    # we need to implement arbitrary dirichlet boundary conditions.
    # Honestly, for intergration, it would probably just be easier to integrate the 
    # interpolation of the function on uniform grid.
    dim = len(funcArr.shape)
    if dim == 1:
        I = np.dot(wArr,funcArr)
    if dim == 2: 
        print('its 2d')
        I = np.einsum('ij,i,j->',funcArr,wArr[0],wArr[1])
    if dim == 3:
        I = np.einsum('ijk,i,j,k->',funcArr,wArr[0],wArr[1],wArr[2])
    if dim == 4:
        I = np.einsum('ijkl,i,j,kl->',funcArr,wArr[0],wArr[1],wArr[2],wArr[3])
    return I

def extend_func(fArr,splOrder,bndry_vals):
    #Assumes the first two elements of bndry_vals are the physical boundary 
    #conditions. This function takes the boundary value and repeats the value for all the
    # fake collocation points that had to be added on each side of the domain.
    if splOrder == 3:
        diff  = 2
    else:
        diff = splOrder - 1 
    zeros = np.zeros(diff)
    zeros[0] = bndry_vals[0]
    zeros[1] = bndry_vals[1]
    zeros[2:2 + int(diff/2)-1] = bndry_vals[0]
    zeros[2+int(diff/2)-1:] = bndry_vals[1]

    fExtend = np.concatenate((fArr,zeros))
    return fExtend

def inv_affine(baseInterval,targetInterval,Pnts):
    '''
    Performs an inverse affine transformation out of the interval [c,d].

    x = a x' + b

    f : [c,d] -> [a,b]
    Parameters
    ----------
    Pnts : ndarray
        array of points to transform

    Returns
    -------
    Pnts : ndarray
          transformed points

    '''
    c = baseInterval[0]
    d = baseInterval[-1]
    a = targetInterval[0]
    b = targetInterval[-1]
    alpha1 = (b - a)/(d-c)
    alpha2 = a - ((b - a)/(d-c))*c
    Pnts_mapped = Pnts*alpha1 + alpha2
    #dgdy = alpha1
    return Pnts_mapped
def exp_transform(xi,xf,nKnots,params):
    if xi < 0: 
        raise(ValueError('x_i boundary must be > 0 for this transform'))
    c1 = params['c1']
    c2 = params['c2']
    xArr = np.linspace(0,1,nKnots)
    knots_transform = c1*(np.exp(c2*np.abs(xArr)) - 1)
    baseInterval = [knots_transform[0],knots_transform[-1]]
    targetInterval = [xi,xf]
    knots = inv_affine(baseInterval,targetInterval,knots_transform)
    #knots = knots_transform*alpha1 + alpha2
    return knots
def arcsin_transform(xi,xf,nKnots,params):
    beta = params['beta']
    xArr = np.linspace(-1,1,nKnots)
    if beta <= 0.0:
        knots_transform = xArr
    else:
        knots_transform = np.arcsin((beta)*xArr)/np.arcsin(beta)
    baseInterval = [knots_transform[0],knots_transform[-1]]
    targetInterval = [xi,xf]
    knots = inv_affine(baseInterval,targetInterval,knots_transform)
    return knots

def arctan_transform(xi,xf,nKnots,params):
    xArr = np.linspace(-1,1,nKnots)
    beta = params['beta']
    knots_transform = 2*np.tan(np.pi*xArr/2)
    baseInterval = [knots_transform[0],knots_transform[-1]]
    targetInterval = [xi,xf]
    knots = inv_affine(baseInterval,targetInterval,knots_transform)
    return knots
def uniform(xi,xf,nKnots):
    knots = np.linspace(xi,xf,nKnots)
    return knots