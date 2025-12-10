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
gB = 1.0
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
z_bndry = (-10,10)
###############################################################################
# Global B-spline parameters
splOrder = 9 #aka, k 
N_knots = 20
knotFunc_params = {'beta': 0.99}

###############################################################################
# Domain boundaries
###############################################################################
bndry_vals_x = [0.0,0.0]
bndry_vals_y = [0.0,0.0]
bndry_vals_z = [0.0,0.0]
###############################################################################
# Generate Basis
###############################################################################
knotArr_x, delta_x_knots = bspline.getKnots(x_bndry[0],x_bndry[1],N_knots,splOrder,
                                          knotFunc = bspline.arcsin_transform,knotFunc_params=knotFunc_params,
                                          extensionMethod='extension')

knotArr_y, delta_y_knots = bspline.getKnots(y_bndry[0],y_bndry[1],N_knots,splOrder,
                                          knotFunc = bspline.arcsin_transform,knotFunc_params=knotFunc_params,
                                          extensionMethod='extension')

knotArr_z, delta_z_knots = bspline.getKnots(z_bndry[0],z_bndry[1],N_knots,splOrder,
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
# Generate z-collocation points
###############################################################################
cPnts_z,bndry_pnts_z = bspline.getCPnts(splOrder,knotArr_z,z_bndry)
B_z = bspline.BMatrix(splOrder,knotArr_z,cPnts_z,bndry='dirichlet',multiprocessing = True)

# Generate the beta matrix that will contain the boundary conditions 
# and construct the B_tilde by appending beta onto B
K_matrix_z = bspline.getKmatrix(splOrder,bndry='dirichlet')
B_tilde_z,beta_z = bspline.betaMatrix(splOrder,knotArr_z,B_z,K_matrix_z,bndry_pnts_z,bndry='dirichlet',multiprocessing = True)
C_tilde_z = bspline.getCtilde(B_tilde_z,splOrder,bndry='dirichlet',bndryVals=bndry_vals_z)

###############################################################################
# Generate derivative matrices for x,y,tau
###############################################################################
Dxx = bspline.getDMatrix(2,splOrder,knotArr_x,cPnts_x,C_tilde_x,bndry='dirichlet')
weights_x = bspline.getWeights(knotArr_x,C_tilde_x,splOrder,bndry='dirichlet')

Dyy = bspline.getDMatrix(2,splOrder,knotArr_y,cPnts_y,C_tilde_y,bndry='dirichlet')
weights_y = bspline.getWeights(knotArr_y,C_tilde_y,splOrder,bndry='dirichlet')

Dzz = bspline.getDMatrix(2,splOrder,knotArr_z,cPnts_z,C_tilde_z,bndry='dirichlet')
weights_z = bspline.getWeights(knotArr_z,C_tilde_z,splOrder,bndry='dirichlet')


class H_HO_operator(LinearOperator):
    """
    Harmonic trap Hamiltonian in 3D position space

    V = g(x^2 + y^2 + z^3) + g/2 \sigma \cdot \vec{B}

    """

    def __init__(self, Dxx, Dyy, Dzz, Nx, Ny, Nz, cPnts_x, cPnts_y, cPnts_z, g, gB,Bvec):
        self.Dxx = Dxx
        self.Dyy = Dyy
        self.Dzz = Dzz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.g = g
        self.gB = gB
        self.Bvec = Bvec
        self.shape = (2*Nx * Ny * Nz, 2*Nx * Ny * Nz)
        self.dtype = np.complex128
        # precompute the potential
        self.x = cPnts_x
        self.y = cPnts_y
        self.z = cPnts_z 
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)
        self.V = self.g*(self.X**2 + self.Y**2 + self.Z**2)
        super().__init__(shape=self.shape, dtype=self.dtype)

    def _matvec(self, psi):
        """
        psi is the vectorized wavefunction
        T = (Dxx + Dyy + Dzz)

        V = g(x^2 + y^2 + z^2) + g/2 \sigma \cdot \vec{B}
        """
        psi = psi.reshape((2,self.Nx, self.Ny, self.Nz))
        Bvec = self.Bvec.reshape(3,self.Nx,self.Ny,self.Nz)
        Hpsi = np.zeros((2,self.Nx, self.Ny,self.Nz),dtype=self.dtype)
        
        Hpsi[0] = -1.0*np.einsum('ji,lki->lkj',self.Dxx,psi[0]) - np.einsum('ji,lik->ljk',self.Dyy,psi[0]) - np.einsum('ji,ilk->jlk',self.Dzz,psi[0]) \
        + self.V * psi[0] + 0.5*gB*Bvec[2]*psi[0] + 0.5*gB*(Bvec[0] - 1j*Bvec[1])*psi[1] 
        
        Hpsi[1] = -1.0*np.einsum('ji,lki->lkj',self.Dxx,psi[1]) - np.einsum('ji,lik->ljk',self.Dyy,psi[1]) - np.einsum('ji,ilk->jlk',self.Dzz,psi[1]) \
        + self.V * psi[1] - 0.5*gB*Bvec[2]*psi[1] + 0.5*gB*(Bvec[0] + 1j*Bvec[1])*psi[0]
        return Hpsi.ravel()

    def _rmatvec(self, psi):
        # Hermitian operator
        return self._matvec(psi)

    def _adjoint(self):
        # Hermitian operator
        return self
#Bvector should be a vector of the form B = (B_x(x,y,z), B_y(x,y,z), B_z(x,y,z)). Should be an array that is (3,Nx*Ny*Nz,Nx*Ny*Nz)
Bvec = np.zeros((3,len(cPnts_x),len(cPnts_y),len(cPnts_z)))
Bvec[2].fill(2.0)
H_HO = H_HO_operator(Dxx,Dyy,Dzz,len(cPnts_x),len(cPnts_y),len(cPnts_z),cPnts_x,cPnts_y,cPnts_z,g,gB,Bvec)
time_start = time.time()
evals, evects = sci.sparse.linalg.eigs(H_HO,k=5,which='SR')
time_end = time.time()
print('diagonalization time (s): ', time_end - time_start)
print(evals)
print(3*np.sqrt(2)/2.0)

