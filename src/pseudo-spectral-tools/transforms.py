import numpy as np

def radial_to_3d(funcArr, cPnts, N):
    cPnts = np.arange(-cPnts[-1],cPnts[-1],.05)
    xx, yy, zz = np.meshgrid(cPnts, cPnts, cPnts, indexing='ij')
    r = np.sqrt(xx**2 + yy**2 + zz**2)
        
    dr = cPnts[-1]/ (len(funcArr) - 1)
    
    indices_3d = np.round(r / dr).astype(int)
    max_index = len(funcArr) - 1
    indices_3d = np.clip(indices_3d, 0, max_index)

    f_xyz = funcArr[indices_3d]

    return f_xyz




class coord_transforms:
    def __init__(self,currentInterval,targetInterval):
        self.a = min(targetInterval)
        self.b = max(targetInterval)
        self.c = min(currentInterval)
        self.d = max(currentInterval)

    def affine(self,Pnts):
        alpha1 = self.b - self.a
        alpha2 = self.d - self.c
        newPnts = (alpha1/alpha2)*(Pnts - self.c) + self.a
        jacobian = alpha1/alpha2
        return newPnts, jacobian
    def affine_old(self,Pnts):
        '''
        performs an affine transformation of the coordinate variables to the
        inteval [-1,1] of the form
        x' = a x + b
        f : [a,b] -> [-1,1]
        Parameters
        ----------
        Pnts : ndarray
            array of points to transform

        Returns
        -------
        Pnts : ndarray
              transformed points

        '''
        alpha1 = .5*(self.b-self.a)
        alpha2 = .5*(self.a+self.b)
        Pnts = Pnts/alpha1  - alpha2/alpha1
        dgdy = 1.0/alpha1

        return Pnts,dgdy
    def inv_affine_old(self,Pnts):
        '''
        Performs an inverse affine transformation out of the interval [-1,1].

        x = a x' + b

        f : [-1,1] -> [a,b]
        Parameters
        ----------
        Pnts : ndarray
            array of points to transform

        Returns
        -------
        Pnts : ndarray
              transformed points

        '''
        alpha1 = .5*(self.b-self.a)
        alpha2 = .5*(self.a+self.b)
        Pnts_mapped = Pnts*alpha1 + alpha2
        dgdy = alpha1
        return Pnts_mapped,dgdy
    def inv_affine_gen(self,Pnts):
        '''
        Performs an inverse affine transformation out of the interval [c,d].

        x = a x' + b

        f : [c,d] -> [a,b]
        Parameters
        ----------
        Pnts : ndarray
            array of points to transform

        Returns
        -------
        Pnts : ndarray
              transformed points

        '''
        alpha1 = (self.b - self.a)/(self.d-self.c)
        alpha2 = self.a - ((self.b - self.a)/(self.d-self.c))*self.c
        Pnts_mapped = Pnts*alpha1 + alpha2
        dgdy = alpha1
        return Pnts_mapped,dgdy
    def arctanh(self,Pnts,L):
        '''
        Performs a arctanh transformation of coordinates on the interval [-1,1].
        This is necessary if we have boundary conditions at x = +/- \inf.

        f: [-1,1] -> (-inf,inf)
        Parameters
        ----------
        Pnts : ndarray
            array of points to transform
        L: float
            length scale of the mapping
        Returns
        -------
        Pnts : ndarray
              transformed points

        '''
        Pnts_mapped = L* np.arctanh(Pnts)
        dgdy = L/(1 + Pnts**2)
        return Pnts_mapped,dgdy
    def cotTransform(self,Pnts,L):
        '''
        Coordinate transform that maps points from [0,2pi]. Taken from Spectral Methods
        in Fluid Dynamics (1988)

        this function maps f: [0,2pi]-> (-inf,inf)
        Cpnts must be in the interval [0,2pi].
        Parameters
        ----------
        Cpnts : ndarray
            grid points to transform.
        L : float
            parameter to control the grid spacing scale.

        Returns
        -------
        result : ndarray
            transformed grid points

        '''
        Pnts_mapped = -L* 1.0/np.tan(.5*Pnts)
        dgdy = .5*L/(np.sin(.5*Pnts)**2)
        return Pnts_mapped,dgdy
    def arcTransform(self,Pnts,beta):
        '''
        Transformation for Chebyshev Gauss_Lobatto points. Transformation taken from
        Kosloff and Tal-Ezer (1991)
        this function maps f: [-1,1]-> [-1,1]
        Cpnts must be in the interval [-1,1].
        Parameters
        ----------
        Cpnts : ndarray
            grid points to transform.
        beta : float
            parameter to control the grid spacing.
        Returns
        -------
        Cpnts_mapped : ndarray
            transformed grid points
        dgdy: derivative of the coordinate transformation with respect to y (Cpnts)
        '''
        beta = float(beta)
        if beta == 0.0:
            Cpnts_mapped = Pnts
            dgdy = np.full(Pnts.shape,1.0)
        else:
            Cpnts_mapped = np.arcsin((beta)*Pnts)/np.arcsin(beta)
            dgdy = (beta/np.arcsin(beta))* 1.0/np.sqrt(1 - (beta*Pnts)**2 )
        return Cpnts_mapped, dgdy

