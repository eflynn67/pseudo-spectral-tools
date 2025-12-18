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


NArr = [20,30,40,50,60,70,80,90,100,110,120]
D1_L2 = np.zeros(len(NArr))
D2_L2 = np.zeros(len(NArr))
indef_L2 = np.zeros(len(NArr))
def_L2 = np.zeros(len(NArr)) 
cusp_L2 = np.zeros(len(NArr))
for i,N in enumerate(NArr):
    ###############################################################################
    # Domain boundaries
    ###############################################################################
    x_bndry = (-10,10)

    ###############################################################################
    # Global B-spline parameters
    splOrder = 5 #aka, k 
    N_knots = N
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
    B = bspline.BMatrix(splOrder,knotArr,cPnts,bndry='dirichlet',multiprocessing = False)

    # Generate the beta matrix that will contain the boundary conditions 
    # and construct the B_tilde by appending beta onto B
    K_matrix = bspline.getKmatrix(splOrder,bndry='dirichlet')
    B_tilde,beta = bspline.betaMatrix(splOrder,knotArr,B,K_matrix,bndry_pnts,bndry='dirichlet',multiprocessing = False)
    C_tilde = bspline.getCtilde(B_tilde,splOrder,bndry='dirichlet',bndryVals=bndry_vals_x)


    weights = bspline.getWeights(knotArr,C_tilde,splOrder,bndry='dirichlet')

    D1 = bspline.getDMatrix(1,splOrder,knotArr,cPnts,C_tilde,bndry='dirichlet')
    D2 = bspline.getDMatrix(2,splOrder,knotArr,cPnts,C_tilde,bndry='dirichlet')

    fArr = test_func(cPnts)
    print(f'D1 anti-hermitian check: {np.linalg.norm(D1 + D1.T.conj())}')

    dfArr = np.einsum('ij,j->i',D1,fArr)
    D1_L2[i] = np.linalg.norm(test_func_D1(cPnts)-dfArr)
    print(f'D1 L2 Error: {np.linalg.norm(test_func_D1(cPnts)-dfArr)}')
    plt.plot(cPnts,test_func_D1(cPnts))
    plt.plot(cPnts,dfArr)
    plt.show()

    print(f'D2 Hermitian Check: {np.linalg.norm(D2 - D2.T.conj())}')

    d2fArr = np.einsum('ij,j->i',D2,fArr)
    D2_L2[i] = np.linalg.norm(test_func_D2(cPnts)-d2fArr)
    print(f'D2 L2 Error: {np.linalg.norm(test_func_D2(cPnts)-d2fArr)}')
    integral = bspline.integrate(np.exp(-cPnts**2),weights)
    exact_integral = np.sqrt(np.pi)
    print('='*50)
    print(f'Infinite Integration')
    print('='*50)
    print(f'Numerical Integral: {integral}')
    print(f'Exact Integral: {exact_integral}')
    indef_L2[i] = np.abs(integral - exact_integral)
    print(f'Integral Difference: {np.abs(integral - exact_integral)}')

    '''
    print('='*50)
    print(f'Finite Integration on interval')
    print('='*50)
    a = 1.49
    b = 2.0


    integral2 = sinc.def_integrate(fArr,cPnts,a,b,h,N,weights)
    exact_integral2 = 0.02657379638979585

    print(f'Numerical Integral2: {integral2}')
    print(f'Exact Integral2: {exact_integral2}')
    print(f'Integral Difference: {np.abs(integral2 - exact_integral2)}')
    def_L2[i] = np.abs(integral2 - exact_integral2)
    '''
    fArr2 = test_func2(cPnts)
    fArr2_D1 = np.einsum('ij,j->i',D1,fArr2)


    cusp_L2[i] = np.linalg.norm(test_func2_D1(cPnts) - fArr2_D1)
    print(f'cusp error: {np.linalg.norm(test_func2_D1(cPnts) - fArr2_D1)}')

plt.plot(NArr,D1_L2,'o',label='D1')
plt.plot(NArr,D2_L2,'o',label='D2')
plt.legend()
plt.yscale('log')
plt.show()


plt.plot(NArr,indef_L2,'o')
plt.yscale('log')
plt.title('Indefinite Integral')
plt.show()

#plt.plot(NArr,def_L2,'o')
#plt.title('Definite Integral')
#plt.yscale('log')
#plt.show()

plt.plot(NArr,cusp_L2,'o')
plt.title('cusp')
plt.yscale('log')
plt.show()
