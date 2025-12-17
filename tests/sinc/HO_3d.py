import numpy as np
import sys
import os
import time
import scipy as sci
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import sinc

###############################################################################
# Summary 
# Solves the static Hartree-fock equations in 2 spatial dimensions.
# Uses scipy packages for sparse lin.alg
# (-0.5\partial_{i}\partial_{i} + V(x,y,z) )\psi_{k} = E_{k} \psi_{k}
# V(x,y,z) is assumed to be some scalar function of x,y,z
###############################################################################

def exact_E(w,nx,ny,nz):
    return(w*(nx + ny + nz + 3.0/2.0))


class H_HO_operator(LinearOperator):
    """
    Harmonic trap Hamiltonian in 3D position space

    V = 0.5*w^2 * (x^2 + y^2 + z^2)

    """

    def __init__(self, Dxx, Dyy, Dzz, Nx, Ny, Nz, cPnts_x, cPnts_y, cPnts_z, w):
        self.Dxx = Dxx
        self.Dyy = Dyy
        self.Dzz = Dzz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.w = w
        self.shape = (Nx * Ny * Nz, Nx * Ny * Nz)
        self.dtype = np.complex128
        # precompute the potential
        self.x = cPnts_x
        self.y = cPnts_y
        self.z = cPnts_z 
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)
        self.V = 0.5*self.w**2*(self.X**2 + self.Y**2 + self.Z**2)
        
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, psi):
        """
        psi is the vectorized wavefunction
        T = -1/2 (Dxx + Dyy + Dzz)

        V = 0.5*w^2 *(x^2 + y^2 + z^2)
        """
        psi = psi.reshape((self.Nx, self.Ny, self.Nz))
        
        return (-0.5*np.einsum('ji,lki->lkj',self.Dxx,psi) - 0.5*np.einsum('ji,lik->ljk',self.Dyy,psi) - 0.5*np.einsum('ji,ilk->jlk',self.Dzz,psi) + self.V * psi).ravel()  
        

    def _rmatvec(self, psi):
        # Hermitian operator
        return self._matvec(psi)

    def _adjoint(self):
        # Hermitian operator
        return self
def exact_E(w,nx,ny,nz):
    return(w*(nx + ny + nz + 1.5))


N = 15
w = 0.5
kappa = 1.0
cPnts, weights = sinc.getCPnts(N,kappa)
h = weights[0]
D2 = sinc.D2(N,h)

H_HO = H_HO_operator(D2,D2,D2,len(cPnts),len(cPnts),len(cPnts),cPnts,cPnts,cPnts,w)
time_start = time.time()
evals, evects = sci.sparse.linalg.eigsh(H_HO,k=32,which='SM')
time_end = time.time()
print('diagonalization time (s): ',time_end - time_start)
print(evals)
print(exact_E(w,0,0,0))
