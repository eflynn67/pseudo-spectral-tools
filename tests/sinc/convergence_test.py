import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import sinc
import transforms
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


NArr = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
D1_L2 = np.zeros(len(NArr))
D2_L2 = np.zeros(len(NArr))
indef_L2 = np.zeros(len(NArr))
def_L2 = np.zeros(len(NArr)) 
cusp_L2 = np.zeros(len(NArr))
for i,N in enumerate(NArr):
    N_total = 2*N + 1
    kappa = 1.0
    cPnts, weights = sinc.getCPnts(N,kappa)
    h = weights[0]
    print(f'Lattice Spacing: h = {h}')
    D1 = sinc.D1(N,h)


    fArr = test_func(cPnts)
    print(f'D1 anti-hermitian check: {np.linalg.norm(D1 + D1.T.conj())}')

    dfArr = np.einsum('ij,j->i',D1,fArr)
    D1_L2[i] = np.linalg.norm(test_func_D1(cPnts)-dfArr)
    print(f'D1 L2 Error: {np.linalg.norm(test_func_D1(cPnts)-dfArr)}')


    D2 = sinc.D2(N,h)
    print(f'D2 Hermitian Check: {np.linalg.norm(D2 - D2.T.conj())}')

    d2fArr = np.einsum('ij,j->i',D2,fArr)
    D2_L2[i] = np.linalg.norm(test_func_D2(cPnts)-d2fArr)
    print(f'D2 L2 Error: {np.linalg.norm(test_func_D2(cPnts)-d2fArr)}')
    integral = sinc.indef_integrate(np.exp(-cPnts**2),weights)
    exact_integral = np.sqrt(np.pi)
    print('='*50)
    print(f'Infinite Integration')
    print('='*50)
    print(f'Numerical Integral: {integral}')
    print(f'Exact Integral: {exact_integral}')
    indef_L2[i] = np.abs(integral - exact_integral)
    print(f'Integral Difference: {np.abs(integral - exact_integral)}')

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

plt.plot(NArr,def_L2,'o')
plt.title('Definite Integral')
plt.yscale('log')
plt.show()

plt.plot(NArr,cusp_L2,'o')
plt.title('cusp')
plt.yscale('log')
plt.show()
