import numpy as np
import sys 
import os
import time
import scipy as sci
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import sinc

def _matvec_func(Dxx,Dyy,Dzz,V,m,psi):
    kin = (-1/(2*m))*(np.einsum('ji,lki->lkj',Dxx,psi) + np.einsum('ji,lik->ljk',Dyy,psi) + np.einsum('ji,ilk->jlk',Dzz,psi)) 
    pot = V*psi
    return kin + pot
class H_op(LinearOperator):
    """
    
    Constructs the HO in 1D for testing
    
    """
    def __init__(self,Dxx,Dyy,Dzz,Nx,Ny,Nz,potential,model_params):
        self.dtype = np.complex128
        self.shape = (Nx * Ny * Nz, Nx * Ny * Nz)
        self.Dxx = Dxx
        self.Dyy = Dyy
        self.Dzz = Dzz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.m = model_params['mass']
        self.V = potential
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, psi):
        psi = psi.reshape((self.Nx, self.Ny, self.Nz))
        return _matvec_func(self.Dxx,self.Dyy,self.Dzz,self.V,self.m,psi).ravel()

    def _rmatvec(self, psi):
        # Hermitian operator
        return self._matvec(psi)
    def _adjoint(self):
        # Hermitian operator
        return self

def exact_evals(numbers_list,lamb):
    evalsArr = np.zeros(len(numbers_list))
    for i,number in enumerate(numbers_list):
        nx = number[0]
        ny = number[1]
        nz = number[2]
        evalsArr[i] = -0.5*((lamb-nx)**(2) + (lamb-ny)**(2) + (lamb-nz)**(2))
    
    return np.sort(evalsArr)


NArr = [10,20,25,30,35,40,45,50,55,60]
principle_numbers = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)] 
k_max = len(principle_numbers)
eval_error = np.zeros((len(NArr),k_max))
times = np.zeros(len(NArr))
for i,N in enumerate(NArr):
    kappa = 1.0
    cPnts, weights = sinc.getCPnts_truncated(N,10)
    #cPnts, weights = sinc.getCPnts(N,kappa)
    h = weights[0]
    D1 = sinc.D1(N,h)
    D2 = sinc.D2(N,h)

    m = 1.0
    Nx = 2*N + 1
    Ny = 2*N + 1
    Nz = 2*N + 1
    g = 0.5
    model_params = {'mass': 1.0}
    lamb = 5.0
    xx, yy , zz = np.meshgrid(cPnts,cPnts,cPnts)
    V = -0.5*lamb*(lamb + 1)*((1.0/np.cosh(xx))**(2) + (1.0/np.cosh(yy))**(2) + (1.0/np.cosh(zz))**(2))
     
    H = H_op(D2,D2,D2,Nx,Ny,Nz,V,model_params)
    start_timer = time.time()
    evals,evects = sci.sparse.linalg.eigsh(H,k=k_max,which='SA')
    end_timer = time.time()
    times[i] = end_timer-start_timer
    print(f'Diagonalization time: {end_timer-start_timer}')
    idx = evals.argsort()
    evals = evals[idx]
    evects = evects[:,idx]
    evects = evects[:,idx]
    exact_PT_evals = exact_evals(principle_numbers,lamb)
    print(evals)
    print(exact_PT_evals)
    error = np.abs(exact_PT_evals - evals)
    plt.scatter(np.full(len(error),N),error)

plt.yscale('log')
plt.show()

plt.plot(NArr,times)
plt.show()
