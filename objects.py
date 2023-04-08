#%%
import numpy as np
import matplotlib.pyplot as plt 
import time
from scipy.special import assoc_laguerre, factorial,genlaguerre
from scipy import fftpack
import scipy.interpolate as interpolate
from statsmodels.stats.weightstats import DescrStatsW

PI=np.pi
"""
higher p modes?? --> solved: gouy phase times (N+1)

if parax and calculated are very different --> maybe a different profile?? Modal decompositon
"""
class Geometry():
    """
    Geometry class, contianing parameters that define the geometry of the optical microcavity.
    """
    def __init__(self,wavelength,L,Rm,pwig,grid_size):

        self.wavelength=wavelength
        self.k=2*PI*wavelength

        self.L=L
        self.Rm=Rm

        self.grid_size=grid_size
        self.pwig=pwig

        self.xmax=xmax=np.sqrt(self.Rm*self.L)/2 # needs to be updated
        self.x = np.linspace(-xmax,xmax,self.grid_size)

    def get_2d_grid(self):
        X,Y=np.meshgrid(self.x,self.x)
        return X,Y

    def mirror_coords(self):  
        """
        Returns the mirror coordinates zm=L-z(x,y).
        """      
        return self.L-self.x**2/(2*self.Rm) - self.x**4/(8*self.Rm**3)*(1-self.pwig)
    
    def mirror_planes(self):
        """
        Returns 4 planes around the mirror surface, used in the interpolation routine in the 'Mode' class.
        """
        zmin = self.L-self.Rm+np.sqrt(self.Rm**2-2*self.xmax**2)
        return np.linspace(zmin,self.L,4)
    
class Field():

    def __init__(self,profile,geometry:Geometry):
        self.geometry=geometry
        self.profile=profile

        self.abs=np.abs(self.profile)
        self.intensity=self.abs**2

        self.phase=np.angle(self.profile)

        self.real=self.profile.real
        self.imag=self.profile.imag
    
    def __mul__(self, other:np.ndarray):
        """
        Operator overload, making a 'Field' object able to
        multiple with a numpy array. Now the profile is multiplied
        with the array
        """
        return self.profile*other
    
class Field1D(Field):
    def __init__(self):
        super().__init__()
        self.phase_on_axis=self.phase[self.geometry.grid_size//2] #extra paramter for 1D field: the on-axis phase

    def power(self):
        """
        Returns the intensity weighted power of the field
        """
        spacing=self.geometry.xmax/self.geometry.grid_size
        weight = abs(self.geometry.x)*2*PI*spacing
        return np.sum(self.intensity*weight)
    
    def phase_moments(self):
        """
        Returns the mean and RMS phase of the field
        """
        weights=self.intensity*abs(self.geometry.x)*PI
        moments = DescrStatsW(self.phase,weights)
        return moments.mean, abs(moments.mean**2 - moments.var)
    
    def phase_shift(self):
        """
        Shifts the field over the average phase
        """
        return Field1D(self*np.exp(-1j*self.phase_moments()[0]),self.geometry)
    
    def phase_plate(self):
        """
        Phase correction for propagation to the z=L plane
        """
        return Field1D(self*np.exp(1j*(self.geometry.L-self.geometry.k*self.geometry.mirror_coords())),self.geometry)

class Field2D(Field):

    def power(self):
        """
        Returns the intensity weighted power of the field
        """
        spacing = 2*self.geometry.xmax/self.geometry.grid_size
        return np.sum(self.intensity.flatten()*spacing**2)

    def cross_section(self):
        """
        Returns a 1D field, containing the profile cross-sected at y=0.
        """
        return Field1D(self.profile[self.geometry.grid_size//2],self.geometry)
    
class Mode():
    """Class containing paraxial eigenmodes of the cavity (LG basis)."""

    def __init__(self,p,l,geometry:Geometry,paraxial):
        self.p=p
        self.l=l
        self.N=2*p+l
        self.geometry=geometry # geometry paramters in which the mode exists
        self.paraxial=paraxial # paraxial or non-paraxial beam propagation

        self.z0 = np.sqrt( self.geometry.L*(self.geometry.Rm-self.geometry.L) )
        self.w0 = np.sqrt(self.z0*self.geometry.wavelength/PI)

    def field_profile(self,z=0):
        """
        Returns a normalized LG(p,l) mode, as a 2D field profile.
        """
        X,Y=self.geometry.get_2d_grid()
        normalization = (-1)**self.p* np.sqrt(factorial(self.p)/(PI*factorial(self.p+self.l)) )# mode normalization

        wz=self.w0 * np.sqrt( 1 + (z/self.z0)**2 )
        scaling = np.sqrt(2)/wz # coordinate scaling

        rho = scaling*np.sqrt(X**2+Y**2) 
        theta=np.arctan2(Y,X)

        f = (rho)**(self.l)*assoc_laguerre(rho**2,self.p,self.l) * np.exp(-rho**2/2)
        return Field2D(f*normalization*scaling,self.geometry)

    def get_gouy(self,z):
        return np.arctan2(z,self.z0)*(self.N+1)
    
    def angular_spectrum(self,z):
        kbar=self.geometry.k/(2*PI) 
        psi_fourier = fftpack.fft2(self.field_profile().profile)
        kbar_x = fftpack.fftfreq(self.geometry.grid_size,2*self.geometry.xmax/self.geometry.grid_size)
        kxx,kyy=np.meshgrid(kbar_x,kbar_x)

        if self.paraxial: # Paraxial propagation, first  order Taylor expansion
            kbar_z = kbar-(kxx**2+kyy**2)/(2*kbar)
        else:  # Non paraxial propagation, all higher order terms are taken into account by the sqrt
            kbar_z = np.sqrt(kbar**2-kxx**2-kyy**2,dtype=complex)

        prop_vector = np.exp(1j*(kbar_z-kbar)*2*PI*z)

        return fftpack.ifft2((psi_fourier*prop_vector))

    def propagate_mirr(self):
        """Returns the electric field at the mirror surface"""
        z_planes = self.geometry.mirror_planes()
        psi_z=np.empty((self.geometry.grid_size,len(z_planes)),dtype=complex)

        for i,plane in enumerate(z_planes):
            psi_z[:,i]=self.angular_spectrum(plane)[:,self.geometry.grid_size//2] # cross-section of the propagated field

        interp_fcn = interpolate.interp1d(z_planes,psi_z,kind='cubic')
        electric_field = np.diag( interp_fcn(self.geometry.mirror_coords()) ) * np.exp(1j*self.geometry.k*(self.geometry.mirror_coords()))
        return Field1D( electric_field, self.geometry )
    
    def propagate_plane(self):
        """Returns the electric field at the z=L plane"""
        electric_field = self.angular_spectrum(self.geometry.L)[self.geometry.grid_size//2]*np.exp(1j*self.geometry.k*self.geometry.L)
        return Field1D(electric_field,self.geometry)