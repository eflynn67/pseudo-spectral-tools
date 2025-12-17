import numpy as np
import sys
import os
import time
import scipy as sci
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import sinc


class H_HO_operator(LinearOperator):
    """
    Harmonic trap Hamiltonian in 2D position space

    V = g(x^2 + y^2 )

    """

    def __init__(self,Dxx,Dyy,Nx, Ny,cPnts_x,cPnts_y,w):
        self.Dxx = Dxx
        self.Dyy = Dyy
        self.Nx = Nx
        self.Ny = Ny
        self.w = w
        self.shape = (Nx * Ny, Nx * Ny)
        self.dtype = np.complex128
        # precompute the potential
        self.x = cPnts_x
        self.y = cPnts_y
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.V = 0.5*self.w**(2)*(self.X** 2 + self.Y** 2 )
        
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, psi):
        """
        psi is the vectorized wavefunction
        T = (Dxx + Dyy)

        V = g(x^2 + y^2)
        """
        psi = psi.reshape((self.Nx, self.Ny))
        return (-0.5*np.einsum('ji,ki->kj',self.Dxx,psi) - 0.5*np.einsum('ji,ik->jk',self.Dyy,psi) + self.V * psi).ravel()  

    def _rmatvec(self, psi):
        # Hermitian operator
        return self._matvec(psi)

    def _adjoint(self):
        # Hermitian operator
        return self
def exact_E(w,nx,ny):
    return(w*(nx + ny + 1.0))



N = 20
w = 0.5
kappa = 1.0
cPnts, weights = sinc.getCPnts(N,kappa)
h = weights[0]
D2 = sinc.D2(N,h)

H_HO = H_HO_operator(D2,D2,len(cPnts),len(cPnts),cPnts,cPnts,w)
time_start = time.time()
evals, evects = sci.sparse.linalg.eigsh(H_HO,k=12,which='SM')
time_end = time.time()

print('diagonalization time (s): ', time_end - time_start)
print(evals)
print(exact_E(w, 0, 0))
psi = np.real(evects[:,0].reshape(len(cPnts),len(cPnts)))

xx, yy = np.meshgrid(cPnts,cPnts)

fig, ax = plt.subplots()
cf = ax.contourf(xx,yy,np.real(psi),extend='both',cmap='Spectral_r',levels=100)
ax.contour(xx,yy,np.real(psi),levels=10,colors='black')
cbar = fig.colorbar(cf)
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.show()
