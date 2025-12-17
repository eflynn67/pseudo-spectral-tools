import numpy as np
import scipy as sci
def sinc(xArr,i,h):
    func = np.sinc((xArr - i*h)/h)
    return func

def getCPnts(N,kappa):
    h = np.sqrt((np.pi/(kappa*(N))))
    jArr = np.arange(-N,N+1,1)
    cPnts = jArr*h
    weights = np.full(len(cPnts),h)
    return cPnts,weights

def getCPnts_truncated(N,x_eff):
    h = x_eff/N
    jArr = np.arange(-N,N+1,1)
    cPnts = jArr*h
    weights = np.full(len(cPnts),h)
    return cPnts, weights  


def get_defWeights(a,b,N,h,iArr,dx=10**(-5)):
    weights = np.zeros(N)
    xArr = np.arange(a,b+dx,dx)
    # shift the indices from i = -N to i = N to j = 0 to j = 2N by setting i = j - N
    for j in iArr:
        weights[i] = sci.integrate.trapezoid(sinc(xArr,(j-N),h),xArr,dx=dx)
    return weights


def D1(N,h): 
    D1 = np.zeros((2*N+1,2*N+1),dtype=np.float64)
    for i in np.arange(0,2*N +1,1):
        for j in np.arange(0,2*N + 1,1):
            if i != j:
                D1[i][j] = ((-1.0)**(i -j))/(h*(i -j))
    return D1
             
def D2(N,h):
    D2 = np.zeros((2*N+1,2*N+1),dtype=np.float64)
    for i in np.arange(0,2*N + 1,1):
        for j in np.arange(0,2*N + 1,1):
    
            if i == j:
                D2[i][i] = -np.pi**(2) / (3.0*h**2)   
            else:
                D2[i][j] = -2.0 * (-1.0)**(i - j) /(h*(i-j))**2
    return D2


def indef_integrate(funcArr,wArr):
    dim = len(funcArr.shape)
    if dim == 1:
        I = np.einsum('i,i ->',funcArr,wArr)
    if dim == 2:
        I = np.einsum('ij,i,j->',funcArr,wArr[0],wArr[1])
    if dim == 3:
        I = np.einsum('ijk,i,j,k->',funcArr,wArr[0],wArr[1],wArr[2])
    if dim == 4:
        I = np.einsum('ijkl,i,j,kl->',funcArr,wArr[0],wArr[1],wArr[2],wArr[3])
    return I 
def def_integrate(funcArr,cPnts,a,b,h,N,wArr):
    dim = len(funcArr.shape)
    if dim == 1:
        u,jac,inv_jac = DE_transform(cPnts,a,b)
        fArr_transformed = interpolate(funcArr,h,N,u)
        I = np.einsum('i,i ->',fArr_transformed*inv_jac,wArr)
    if dim == 2:
        I = np.einsum('ij,i,j->',funcArr,wArr[0],wArr[1])
    if dim == 3:
        I = np.einsum('ijk,i,j,k->',funcArr,wArr[0],wArr[1],wArr[2])
    if dim == 4:
        I = np.einsum('ijkl,i,j,kl->',funcArr,wArr[0],wArr[1],wArr[2],wArr[3])
    return I



def interpolate(fArr,h,N,grid):
    card = np.zeros((2*N+1,len(grid)))
    for j in range(2*N +1 ):
        card[j] = sinc(grid,j - N,h)
    result = np.einsum('i,ij->j',fArr,card)
    return result

def tanh(xArr,a,b):
    u = 0.5*(b - a)*np.tanh(xArr*0.5) + 0.5*(b + a)
    # jacobian = du/dx and inv_jacobian = dx/du
    jacobian = 1/(b - u) + 1/(u - a) 
    inv_jacobian = 0.25*(b-a)* 1.0/np.cosh(xArr*0.5)**2
    return u,jacobian,inv_jacobian

def DE_transform(xArr,a,b):
    u = 0.5*(b-a)*np.tanh(0.5*np.pi*np.sinh(xArr)) + 0.5*(b+a)
    cosh_arg = 0.5*np.pi*np.sinh(xArr)
    ## removing values greater than 709.7, the maximum for stable evaluation of cosh from numpy
    inf_vals_condition = np.abs(cosh_arg) > 709.7
    # just set the values above the threshold to infinity. These numbers don't matter anyway, they contribute 0 to the integral
    cosh_arg[inf_vals_condition]= np.inf
    cosh_part = np.cosh(cosh_arg)
    inv_jacobian = 0.25*np.pi*(b-a)*np.cosh(xArr)*(1.0/cosh_part)**(2)
    jacobian = None 
    return u,jacobian,inv_jacobian

'''
def __radial_to_3d_spherical(fArr,N_theta,N_phi):
    
    Converts a function with spherical symmetry f(r) to a function of spherical coordinates f(r,theta,phi) by rotation.
    
    theta : [0,2pi]
    phi: [0,pi]
    
    thetaArr = np.linalg(0,2*np.pi,N_theta)
    phiArr = np.linalg(0,2*np.pi,N_phi)
    fArrNew = np.zeros((N_phi,N_theta,len(fArr))
    for i in range(N_phi):
        for j in range(N_theta):
            fArrNew[i][j] = fArr
    return None
def interpolate_spherical(fArr,thetaPnts,phiPnts):
    
    Interpolates a function of radial coordinate f(r) to the sinc grid in cartensian (x,y,z) coordinates

    
    return None
'''
