"""
Simulations of optical microcavities - Fourier Propagation

Last edit: 20 Mar 2023 16:00

2D Fourier propagaion of Gaussian laser modes - subroutines

Idea to work out further:
    all the parameters in the beginning, could be poured into an object, so they
    are easily accessible to other routines.

To do:
    look for non-paraxial effects in the imaginary part of psi_interp !!
    --> PAPER CORNE
    check Gouy phase for z=L mode
"""
#%%
import numpy as np
import matplotlib.pyplot as plt 
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import assoc_laguerre, eval_hermitenorm, factorial,genlaguerre
from scipy import fftpack
import scipy.interpolate as interpolate
from statsmodels.stats.weightstats import DescrStatsW

PI = np.pi
#%%
# Normalized Gaussian mode functions


def LG(p,l,X,Y,w):
    """
    Input: 
        Mode numbers p and l
        2D cartesian coordinates X and Y (meshgrid)
        Beam width at z=0
    --------------------------------------
    Output:
        Normalized (l,p) Laguerre-Gauss mode
    """
    # l=abs(n-m)
    # p=min((n,m))
    normalization = (-1)**p* np.sqrt(factorial(p)/(PI*factorial(p+l)) )
    scaling = np.sqrt(2)/w
    rho = scaling*np.sqrt(X**2+Y**2)
    theta=np.arctan2(Y,X)

    f = (rho)**(l)*assoc_laguerre(rho**2,p,l) * np.exp(-rho**2/2)
    return f*normalization*scaling#*np.exp(1j*theta*l) # vector modes!!, later
def HG(n,m,X,Y,w):

    """
    Input: 
        Mode numbers n and m
        2D cartesian coordinates X and Y (meshgrid)
        Beam width at z=0
    --------------------------------------
    Output:
        Normalized (n,m) Hermite-Gauss mode
    """

    N=n+m
    scaling = np.sqrt(2)/w
    normalization = np.sqrt(2/PI) * 1/(np.sqrt(factorial(n)*factorial(m))) * 2**(-N/2)
    return normalization*eval_hermitenorm(n,scaling*X)*eval_hermitenorm(m,scaling*Y)*fundamental(X,Y,np.sqrt(2)*w)

# Calculation of field propreties
def intensity(psi):
    """
    Input:
        psi: 1D or 2D field profile
    
    Output:
        Field intensty
    """
    return np.abs(psi)**2
def calc_phase(psi):
    """
    Input:
        psi: 1D or 2D field profile
    
    Output:
        Field phase
    """
    return np.angle(psi)

def phase_moments(psi,x):
    """
    Input:
        psi: 1D field profile

    Output:
        mean and std of the phase
    """
    weights=intensity(psi)*x*PI
    moments = DescrStatsW(calc_phase(psi),weights)
    return moments.mean, np.sqrt(moments.var)

def phase_shift(psi,x):
    """
    Input:
        psi: 1D field profile

    Output:
        1D field phase shifted over the average phase
    """
    return psi*np.exp(-1j*phase_moments(psi,x)[0])

def power(Psi,x_spacing):
    """
    Input:
        Psi: 2D field profile
        x: array of x elements, needed for the weight of a 1D power function
    Output:
        Field power, as a sum over the intensity times the grid spacing,
        if the field is 1D, a 2*PI*x 
    """
    return np.sum(intensity(Psi.flatten())*x_spacing**2)
def power_1d(psi,x):
    """
    Input:
        Psi: 1D field profile
        x: array of x elements, needed for the weight of a 1D power function
    Output:
        Field power, as a sum over the intensity times the Jacobian
        for polar coordinates
    """
    x_spacing = 2*max(x)/len(x)
    return np.sum(intensity(psi)*x*PI*x_spacing)

def Fourier2D_scipy_pack(Psi_0,x,z,wavelength=1,paraxial=False):
    """
    Fourier function using scipy module, faster computation
    Input:
        Psi_0: 2D Slowly-varying field at z=0
        x: 1D x coordinate, to determine kx values
        z: z coordinate to which to propagate to
    ---------------------------------------------

    Output:
        2D Propagated, slowly-varying field at z
    """
    kbar=1/wavelength
    psi_fourier = fftpack.fft2(Psi_0)
    kbar_x = fftpack.fftfreq(len(x),2*max(x)/len(x)) # domain goed from negative to positive, so 2*maximal value !!
    kxx,kyy=np.meshgrid(kbar_x,kbar_x)

    if paraxial: # Paraxial propagation, first  order Taylor expansion
        kbar_z = kbar-(kxx**2+kyy**2)/(2*kbar)
    else:  # Non paraxial propagation, all higher order terms are taken into account by the sqrt
        kbar_z = np.sqrt(kbar**2-kxx**2-kyy**2,dtype=complex)

    prop_vector = np.exp(1j*(kbar_z-kbar)*2*PI*z)
    return fftpack.ifft2((psi_fourier*prop_vector))

def interpolate_mirror(Psi_0,x,z_planes,paraxial=False):
    
    """
    Input:
        Psi_0: 2D array, containing the slowly-varying field at z=0 (flat mirror)
        x: 1D grid of x coordinates on a grid
        z_planes: array containing planes around curved mirror to propagate to

    ----------------------------------------------------------------------------
    Output:
        1D interpolation function, which contains a continuous function
        for psi(x,z), cross-sected at y=0
    """

    grid_size=len(x)
    psi_z=np.empty((grid_size,len(z_planes)),dtype=complex)
    for i,plane in enumerate(z_planes):
        psi_z[:,i]=Fourier2D_scipy_pack(Psi_0,x,plane,paraxial=paraxial)[:,grid_size//2]
    
    return interpolate.interp1d(z_planes,(psi_z),kind='cubic')

def fourier_mirror(Psi_0,x,z_planes,paraxial=False):
    """
    Input:
        Psi_0: 2D array, containing the slowly-varying field at z=0 (flat mirror)
        x: 1D grid of x coordinates on a grid
        z_planes: array containing planes around curved mirror to propagate to

    ----------------------------------------------------------------------------
    Output:
        1D interpolation function, which contains a continuous function
        for psi(x,z), cross-sected at y=0
    """
    
    grid_size=len(x)
    psi_z=np.empty((grid_size,len(z_planes)),dtype=complex)
    for i,plane in enumerate(z_planes):
        psi_z[:,i]=Fourier2D_scipy_pack(Psi_0,x,plane,paraxial=paraxial)[:,grid_size//2]
    
    return np.diag(psi_z)

def propagate(Psi_0,x,z,wavelength=1,paraxial=False,mirror=True,z_planes=None):
    """
    Propagates input, slowly-varying field from flat mirror to
        (1) Curved mirror if mirror=True
        (2) z=L plane if mirror=False
    
    Input:
        Psi_0: 2D array, containing the slowly-varying field at z=0 (flat mirror)
        x: 1D grid of x coordinates on a grid
        z: float or array like, z values to propagate to
        wavelength: wavelength parameter, standard is 1 [microns]
        paraxial: boolean variable, paraxial or non-paraxial propagation
        mirror: boolean variable, 
        z_planes: optional, planes for interpolation at mirror
    
    ----------------------------------------------------------------------
    Output:
        Output !Electric! field at
            (1) Curved mirror if mirror=True
            (2) z=L plane if mirror=False
        output field is a 1D cross-section at y=0
    """

    grid_size=len(x)
    k=2*PI/wavelength

    if mirror:
        interp_1d=interpolate_mirror(Psi_0,x,z_planes,paraxial=paraxial)
        Psi_interp = interp_1d(z)
        return np.diag(Psi_interp)*np.exp(1j*k*z)

    else:
        return Fourier2D_scipy_pack(Psi_0,x,z,paraxial=paraxial)[grid_size//2]*np.exp(-1j*k*z)