import numpy as np
import sys 
import os
import time
import scipy as sci
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import sinc

def _matvec_func(Dxx,V,m,psi):
    kin = (-1/(2*m))*np.einsum('ij,j->i',Dxx,psi) 
    pot = V*psi
    return kin + pot
class H_op(LinearOperator):
    """
    
    Constructs the HO in 1D for testing
    
    """
    def __init__(self,Dxx,Nx,potential,model_params):
        self.dtype = np.complex128
        self.shape = (Nx,Nx)

        self.Dxx = Dxx

        self.Nx = Nx

        self.m = model_params['mass']
        self.V = potential
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, psi):
        
        return _matvec_func(self.Dxx,self.V,self.m,psi).ravel()

    def _rmatvec(self, psi):
        # Hermitian operator
        return self._matvec(psi)
    def _adjoint(self):
        # Hermitian operator
        return self

def exact_evals(n,lamb):
    return -0.5*(lamb-n)**(2)


NArr = [10,15,20,25,30,35,40,45,50,55,60]
principle_numbers = np.arange(0,10,1)
k_max = len(principle_numbers)
eval_error = np.zeros((len(NArr),k_max))

for i,N in enumerate(NArr):
    kappa = 1.0
    cPnts, weights = sinc.getCPnts_truncated(N,10)
    #cPnts, weights = sinc.getCPnts(N,kappa)
    h = weights[0]
    D1 = sinc.D1(N,h)
    D2 = sinc.D2(N,h)

    m = 1.0
    Nx = 2*N + 1
    g = 0.5
    model_params = {'mass': 1.0}
    lamb = 5.0
    V = -0.5*lamb*(lamb + 1)*(1.0/np.cosh(cPnts))**(2)
     
    H = H_op(D2,Nx,V,model_params)
    start_timer = time.time()
    evals,evects = sci.sparse.linalg.eigsh(H,k=k_max,which='SA')
    end_timer = time.time()

    print(f'Diagonalization time: {end_timer-start_timer}')
    idx = evals.argsort()
    evals = evals[idx]
    evects = evects[:,idx]
    evects = evects[:,idx]
    exact_PT_evals = exact_evals(principle_numbers,lamb)
    error = np.abs(exact_PT_evals - evals)
    #evals_error[i] = error
    plt.scatter(np.full(len(error),N),error)

plt.yscale('log')
plt.show()
