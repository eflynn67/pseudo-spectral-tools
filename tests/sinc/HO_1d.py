import numpy as np
import sys 
import os
import time
import scipy as sci
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import sinc


N = 10
kappa = 1.0
cPnts, weights = sinc.getCPnts(N,kappa)
print(cPnts)
h = weights[0]
D1 = sinc.D1(N,h)
D2 = sinc.D2(N,h)
def HO_matvec(Dxx,V,m,psi):
    kin = (-1/(2*m))*np.einsum('ij,j->i',Dxx,psi) 
    pot = V*psi
    return kin + pot
class H_HO(LinearOperator):
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
        
        return HO_matvec(self.Dxx,self.V,self.m,psi).ravel()

    def _rmatvec(self, psi):
        # Hermitian operator
        return self._matvec(psi)
    def _adjoint(self):
        # Hermitian operator
        return self

m = 1.0
Nx = 2*N + 1
g = 0.5
model_params = {'mass': 1.0}
V = g* cPnts**2 
 
H = H_HO(D2,Nx,V,model_params)
k_max = 3
start_timer = time.time()
evals,evects = sci.sparse.linalg.eigsh(H,k=k_max,which='SM')
end_timer = time.time()

print(f'Diagonalization time: {end_timer-start_timer}')
idx = evals.argsort()
evals = evals[idx]
evects = evects[:,idx]
evects = evects[:,idx]
print(evals)


