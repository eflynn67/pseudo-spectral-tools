import numpy as np


def getCPnts(N,kappa):
    h = np.sqrt((np.pi/(kappa*N)))
    jArr = np.arange(-N,N+1,1)
    cPnts = jArr*h
    weights = np.full(len(cPnts),h)
    return cPnts,weights

def D1(N,h): 
    D1 = np.zeros((2*N+1,2*N+1),dtype=np.float64)
    for i in range(2*N +1):
        for j in range(2*N + 1):
            if i != j:
                D1[i][j] = (-1)**(i -j)/(h*(i -j))
    return D1
             
def D2(N,h):
    D2 = np.zeros((2*N+1,2*N+1),dtype=np.float64)
    for i in range(2*N + 1):
        for j in range(2*N + 1):
            if i == j:
                D2[i][i] = -np.pi**(2) / (3.0*h**2)   
            else:
                D2[i][j] = -2 * (-1)**(i - j) /(h*(i-j))**2
    return D2


def integrate(funcArr,wArr):
    dim = len(funcArr.shape)
    if dim == 1:
        I = np.einsum('i,i ->',funcArr,wArr)
    if dim == 2:
        print('its 2d')
        I = np.einsum('ij,i,j->',funcArr,wArr[0],wArr[1])
    if dim == 3:
        I = np.einsum('ijk,i,j,k->',funcArr,wArr[0],wArr[1],wArr[2])
    if dim == 4:
        I = np.einsum('ijkl,i,j,kl->',funcArr,wArr[0],wArr[1],wArr[2],wArr[3])
    return I 


