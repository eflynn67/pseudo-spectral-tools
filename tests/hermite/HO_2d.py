import numpy as np
import scipy as sci
from scipy.sparse.linalg import LinearOperator
import sys
import functools
import time
import matplotlib.pyplot as plt
sys.path.insert(0, '../src/')
import bspline

###############################################################################
# Summary 
# Solves the static Harmonic Oscillator equations in 2 spatial dimensions.
# Uses scipy packages for sparse lin.alg
# (-1/2 * \partial_{x}\partial_{x}+ \partial_{y}\partial_{y} + V(x,y) )\psi_{k} = E_{k} \psi_{k}
# V(x,y) is assumed to be some scalar function of x and y
###############################################################################

def exact_E(w,nx,ny):
    return(w*(nx + ny + 1.0))

###############################################################################
# Physical Parameters
###############################################################################
AMax = 1 #number of particles
w = 1.0

###############################################################################
# Domain boundaries
###############################################################################
x_bndry = (-10,10)
y_bndry = (-10,10)

###############################################################################
# Global B-spline parameters
splOrder = 11 #aka, k 
N_knots = 20
knotFunc_params = {'beta': 0.99}

###############################################################################
# Domain boundaries
###############################################################################
bndry_vals_x = [0.0,0.0]
bndry_vals_y = [0.0,0.0]

###############################################################################
# Generate Basis
###############################################################################
knotArr_x, delta_x_knots = bspline.getKnots(x_bndry[0],x_bndry[1],N_knots,splOrder,
                                          knotFunc = bspline.arcsin_transform,knotFunc_params=knotFunc_params,
                                          extensionMethod='extension')

knotArr_y, delta_y_knots = bspline.getKnots(y_bndry[0],y_bndry[1],N_knots,splOrder,
                                          knotFunc = bspline.arcsin_transform,knotFunc_params=knotFunc_params,
                                          extensionMethod='extension')

###############################################################################
# Generate x-collocation points
###############################################################################
cPnts_x,bndry_pnts_x = bspline.getCPnts(splOrder,knotArr_x,x_bndry)
B_x = bspline.BMatrix(splOrder,knotArr_x,cPnts_x,bndry='dirichlet',multiprocessing = True)

# Generate the beta matrix that will contain the boundary conditions 
# and construct the B_tilde by appending beta onto B
K_matrix_x = bspline.getKmatrix(splOrder,bndry='dirichlet')
B_tilde_x,beta_x = bspline.betaMatrix(splOrder,knotArr_x,B_x,K_matrix_x,bndry_pnts_x,bndry='dirichlet',multiprocessing = True)
C_tilde_x = bspline.getCtilde(B_tilde_x,splOrder,bndry='dirichlet',bndryVals=bndry_vals_x)

###############################################################################
# Generate y-collocation points
###############################################################################
cPnts_y,bndry_pnts_y = bspline.getCPnts(splOrder,knotArr_y,y_bndry)
B_y = bspline.BMatrix(splOrder,knotArr_y,cPnts_y,bndry='dirichlet',multiprocessing = True)

# Generate the beta matrix that will contain the boundary conditions 
# and construct the B_tilde by appending beta onto B
K_matrix_y = bspline.getKmatrix(splOrder,bndry='dirichlet')
B_tilde_y,beta_y = bspline.betaMatrix(splOrder,knotArr_y,B_y,K_matrix_y,bndry_pnts_y,bndry='dirichlet',multiprocessing = True)
C_tilde_y = bspline.getCtilde(B_tilde_y,splOrder,bndry='dirichlet',bndryVals=bndry_vals_y)

###############################################################################
# Generate derivative matrices for x,y,tau
###############################################################################
Dxx = bspline.getDMatrix(2,splOrder,knotArr_x,cPnts_x,C_tilde_x,bndry='dirichlet')
weights_x = bspline.getWeights(knotArr_x,C_tilde_x,splOrder,bndry='dirichlet')

Dyy = bspline.getDMatrix(2,splOrder,knotArr_y,cPnts_y,C_tilde_y,bndry='dirichlet')
weights_y = bspline.getWeights(knotArr_y,C_tilde_y,splOrder,bndry='dirichlet')

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

H_HO = H_HO_operator(Dxx,Dyy,len(cPnts_x),len(cPnts_y),cPnts_x,cPnts_y,w)
time_start = time.time()
evals, evects = sci.sparse.linalg.eigs(H_HO,k=5,which='SR')
time_end = time.time()

print('diagonalization time (s): ', time_end - time_start)
print(evals)
print(exact_E(w, 0, 0))
psi = np.real(evects[:,0].reshape(len(cPnts_x),len(cPnts_y)))

xx, yy = np.meshgrid(cPnts_x,cPnts_y)

fig, ax = plt.subplots()
cf = ax.contourf(xx,yy,np.real(psi),extend='both',cmap='Spectral_r',levels=100)
ax.contour(xx,yy,np.real(psi),levels=10,colors='black')
cbar = fig.colorbar(cf)
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.show()
