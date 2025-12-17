import numpy as np
import sys 
import os
import scipy as sci
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
import time
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import bspline

###############################################################################
# Domain boundaries
###############################################################################
x_bndry = (-6,6)

###############################################################################
# Global B-spline parameters
splOrder = 9 #aka, k 
N_knots = 21
knotFunc_params = {'beta': 0.0}

###############################################################################
# Domain boundaries
###############################################################################
bndry_vals_x = [0.0,0.0]

###############################################################################
# Generate Basis
###############################################################################
knotArr, delta_x_knots = bspline.getKnots(x_bndry[0],x_bndry[1],N_knots,splOrder,
                                          knotFunc = bspline.arcsin_transform,knotFunc_params=knotFunc_params,
                                          extensionMethod='extension')

###############################################################################
# Generate x-collocation points
###############################################################################
cPnts,bndry_pnts = bspline.getCPnts(splOrder,knotArr,x_bndry)
B = bspline.BMatrix(splOrder,knotArr,cPnts,bndry='dirichlet',multiprocessing = True)

# Generate the beta matrix that will contain the boundary conditions 
# and construct the B_tilde by appending beta onto B
K_matrix = bspline.getKmatrix(splOrder,bndry='dirichlet')
B_tilde,beta = bspline.betaMatrix(splOrder,knotArr,B,K_matrix,bndry_pnts,bndry='dirichlet',multiprocessing = True)
C_tilde = bspline.getCtilde(B_tilde,splOrder,bndry='dirichlet',bndryVals=bndry_vals_x)

D2 = bspline.getDMatrix(2,splOrder,knotArr,cPnts,C_tilde,bndry='dirichlet')

weights = bspline.getWeights(knotArr,C_tilde,splOrder,bndry='dirichlet')

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
Nx = len(cPnts)
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
