import numpy as np
import sys 
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import sinc

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


N = 50
kappa = 1.0
cPnts, weights = sinc.getCPnts(N,kappa)
h = weights[0]
fArr = test_func(cPnts)

plt.plot(cPnts,fArr)
plt.show()

D1 = sinc.D1(N,h)
print(f'D1 anti-hermitian check: {np.linalg.norm(D1 + D1.T.conj())}')

dfArr = np.einsum('ij,j->i',D1,fArr)
plt.plot(cPnts,test_func_D1(cPnts) - dfArr)
plt.yscale('log')
plt.show()

D2 = sinc.D2(N,h)
print(f'D2 Hermitian Check: {np.linalg.norm(D2 - D2.T.conj())}')

d2fArr = np.einsum('ij,j->i',D2,fArr)

plt.plot(cPnts,test_func_D2(cPnts)-d2fArr)
plt.yscale('log')
plt.show()

integral = sinc.integrate(np.exp(-cPnts**2),weights)
exact_integral = np.sqrt(np.pi)
print(f'Numerical Integral: {integral}')
print(f'Exact Integral: {exact_integral}')
print(f'Integral Difference: {np.abs(integral - exact_integral)}')

fArr2 = test_func2(cPnts)
fArr2_D1 = np.einsum('ij,j->i',D1,fArr2)
plt.plot(cPnts,fArr2)
plt.show()

plt.plot(cPnts,test_func2_D1(cPnts))
plt.plot(cPnts,fArr2_D1)
plt.show()

plt.plot(cPnts,np.abs(test_func2_D1(cPnts) - fArr2_D1))
plt.show()

