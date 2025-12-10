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
# Solves the static Hartree-fock equations in 2 spatial dimensions.
# Uses scipy packages for sparse lin.alg
# (hbar/2mp * \partial_{i}\partial_{i} + V(x,y;\rho) )\psi_{k} = E_{k} \psi_{k}
# V(x,y;\rho) is assumed to be some scalar function of x and y
###############################################################################



###############################################################################
# Physical Parameters
###############################################################################
AMax = 10 #number of particles
g = 0.5
Mp = 938.27208943 # mass of proton in MeV
#hb2m0 is in MeV fm^2 (literally hbar^2 / 2mp in units MeV fm^2)
model_params = {'AMax':AMax,'g_sigma':g,'Mp':Mp,'hb2m0':20.735530}

###############################################################################
# Solver Parameters
###############################################################################
etol = 10**(-2) # tolerence for self-consistent solve 
alpha = 1.0
sovler_params = {'etol':etol,'alpha': 1.0}


###############################################################################
# Domain boundaries
###############################################################################
x_bndry = (-10,10)
y_bndry = (-10,10)

###############################################################################
# Global B-spline parameters
splOrder = 9 #aka, k 
N_knots = 80
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

    def __init__(self,Dxx,Dyy,Nx, Ny,cPnts_x,cPnts_y,g):
        self.Dxx = Dxx
        self.Dyy = Dyy
        self.Nx = Nx
        self.Ny = Ny
        self.g = g
        self.shape = (2*Nx * Ny, 2*Nx * Ny)
        self.dtype = np.complex128
        # precompute the potential
        self.x = cPnts_x
        self.y = cPnts_y
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.V = self.g*(self.X** 2 + self.Y** 2 )
        
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, psi):
        """
        psi is the vectorized wavefunction
        T = (Dxx + Dyy)

        V = g(x^2 + y^2)
        """
        psi = psi.reshape((2,self.Nx, self.Ny))
        Hpsi = np.zeros((2,self.Nx, self.Ny),dtype=self.dtype) 
        for i in range(0,2):
            Hpsi[i] = -1.0*np.einsum('ji,ki->kj',self.Dxx,psi[i]) - np.einsum('ji,ik->jk',self.Dyy,psi[i]) + self.V * psi[i]
        return Hpsi.ravel()  
        #return (self.V * psi).ravel()

    def _rmatvec(self, psi):
        # Hermitian operator
        return self._matvec(psi)

    def _adjoint(self):
        # Hermitian operator
        return self

H_HO = H_HO_operator(Dxx,Dyy,len(cPnts_x),len(cPnts_y),cPnts_x,cPnts_y,g)
time_start = time.time()
evals, evects = sci.sparse.linalg.eigs(H_HO,k=5,which='SR')
time_end = time.time()
print(time_end - time_start)
print(evals)
print(np.sqrt(2))
