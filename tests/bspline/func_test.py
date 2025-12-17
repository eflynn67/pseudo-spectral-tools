import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import bspline
def test_func(xArr):
    return np.exp(-xArr**2)*np.sin(xArr) 
def test_func_D1(xArr):
    return np.exp(-xArr**2)*np.cos(xArr) - 2*np.exp(-xArr**2)*xArr*np.sin(xArr)
def test_func_D2(xArr):
    return np.exp(-xArr**2)*(-4*xArr*np.cos(xArr) + (-3 + 4*xArr**2)*np.sin(xArr))
def test_func2(xArr):
    return np.exp(-np.abs(xArr))
def test_func2_D1(xArr):
    return np.sign(xArr)*np.exp(-np.abs(xArr))

###############################################################################
# Domain boundaries
###############################################################################
x_bndry = (-5,5)

###############################################################################
# Global B-spline parameters
splOrder = 11 #aka, k 
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


weights = bspline.getWeights(knotArr,C_tilde,splOrder,bndry='dirichlet')

D1 = bspline.getDMatrix(1,splOrder,knotArr,cPnts,C_tilde,bndry='dirichlet')
D2 = bspline.getDMatrix(2,splOrder,knotArr,cPnts,C_tilde,bndry='dirichlet')

print(f'D1 anti-hermitian check: {np.linalg.norm(D1 + D1.T.conj())}')

fArr = test_func(cPnts)
dfArr = np.einsum('ij,j->i',D1,fArr)
print(f'D1 L2 Error: {np.linalg.norm(test_func_D1(cPnts)-dfArr)}')

plt.plot(cPnts,np.abs(test_func_D1(cPnts) - dfArr))
plt.title('1st Derivative Error')
plt.yscale('log')
plt.show()

print(f'D2 Hermitian Check: {np.linalg.norm(D2 - D2.T.conj())}')

d2fArr = np.einsum('ij,j->i',D2,fArr)
print(f'D2 L2 Error: {np.linalg.norm(test_func_D2(cPnts)-d2fArr)}')
plt.plot(cPnts,np.abs(test_func_D2(cPnts)-d2fArr))
plt.yscale('log')
plt.title('2nd Derivative Error')
plt.show()

print('='*50)
print(f'Infinite Integration')
print('='*50)

integral = bspline.integrate(np.exp(-cPnts**2),weights)
exact_integral = np.sqrt(np.pi)
print(f'Gaussian Numerical Integral: {integral}')
print(f'Gaussian Exact Integral: {exact_integral}')
print(f'Gaussian Integral Difference: {np.abs(integral - exact_integral)}')

fArr2 = test_func2(cPnts)
fArr2_D1 = np.einsum('ij,j->i',D1,fArr2)
plt.plot(cPnts,fArr2)
plt.show()

plt.plot(cPnts,test_func2_D1(cPnts))
plt.plot(cPnts,fArr2_D1)
plt.show()

plt.plot(cPnts,np.abs(test_func2_D1(cPnts) - fArr2_D1))
plt.show()
print(f'cusp error: {np.linalg.norm(test_func2_D1(cPnts) - fArr2_D1)}')
'''
print('='*50)
print('Interpolation Test')
print('='*50)
newGrid = np.arange(cPnts[0],cPnts[-1]+10**(-3),10**(-3))
fInterp = sinc.interpolate(fArr,h,N,newGrid)
print(fArr.shape)
print(fInterp.shape)
exactf = test_func(newGrid)
print(f'Interpolation L2 error = {np.linalg.norm(fInterp - exactf)}')
plt.plot(newGrid,np.abs(fInterp - exactf))
plt.yscale('log')
plt.show()

print('='*50)
print('Interpolation on sub set of function domain test')
print('='*50)
a = 1.49
b = 2.0
dx_interp = 10**(-3)
newGrid = np.arange(a,b+dx_interp,dx_interp)
fInterp = sinc.interpolate(fArr,h,N,newGrid)
print(fArr.shape)
print(fInterp.shape)
exactf = test_func(newGrid)
print(f'Definite Interval Interpolation L2 Error = {np.linalg.norm(fInterp - exactf)}')

plt.plot(newGrid,np.abs(fInterp - exactf))
plt.yscale('log')
plt.show()


print('='*50)
print('Definite Integration with Interpolation Test')
print('='*50)

a = 1.49
b = 2.0


integral2 = sinc.def_integrate(fArr,cPnts,a,b,h,N,weights)
exact_integral2 = 0.02657379638979585

print(f'Numerical Integral2: {integral2}')
print(f'Exact Integral2: {exact_integral2}')
print(f'Integral Difference: {np.abs(integral2 - exact_integral2)}')
'''
