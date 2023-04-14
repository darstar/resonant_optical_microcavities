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

    def __mul__(self, other):
        """
        Operator overload, making a 'Field' object able to
        multiple with a numpy array, a numer or another Field object.
        """

        if isinstance(other, self.__class__):
            return Field(self.profile*other.profile,self.geometry)
        elif isinstance(other, complex) or isinstance(self, np.ndarray):
            return Field(self.profile*other,self.geometry)

    def electric_field(self,z):
        return self*np.exp(1j*self.geometry.k*z)
    
    def slowly_varying(self,z):
        return self*np.exp(-1j*self.geometry.k*z)
    
class Field1D(Field):
    def __init__(self,profile,geometry:Geometry):
        super().__init__(profile,geometry)
        self.phase_on_axis=self.phase[self.geometry.grid_size//2] #extra paramter for 1D field: the on-axis phase

    def __mul__(self, other):
        """
        Operator overload, making a 'Field' object able to
        multiple with a numpy array, a numer or another Field object.
        """
        
        if isinstance(other, self.__class__):
            return Field1D(self.profile*other.profile,self.geometry)
        
        elif isinstance(other, int) or isinstance(other, float):
            return self.profile*other
        
        elif isinstance(other, np.ndarray) or isinstance(other,complex) or isinstance(other,np.complex128):
            if np.iscomplexobj(other):
                return Field1D(self.profile*other,self.geometry)
            else:
                return self.profile*other
            
    def __add__(self,other):
        return Field1D(self.profile+other.profile,self.geometry)
    
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
        return self*np.exp(-1j*self.phase_moments()[0])
    
    def phase_plate(self):
        """
        Phase correction for propagation to the z=L plane
        """
        return self*np.exp(1j*(self.geometry.L-self.geometry.k*self.geometry.mirror_coords()))

class Field2D(Field):
    def __mul__(self, other):
        """
        Operator overload, making a 'Field' object able to
        multiple with a numpy array, a numer or another Field object.
        """
        
        if isinstance(other, self.__class__):
            return Field2D(self.profile*other.profile,self.geometry)
        
        elif isinstance(other, int) or isinstance(other, float):
            return self.profile*other
        
        elif isinstance(other, np.ndarray) or isinstance(other,complex) or isinstance(other,np.complex128):
            if np.iscomplexobj(other):
                return Field2D(self.profile*other,self.geometry)
            else:
                return self.profile*other
    
    def __add__(self,other):
        return Field2D(self.profile+other.profile,self.geometry)
    
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

    def __init__(self,p,l,geometry:Geometry,paraxial,alphas=None):
        self.p=p
        self.l=l
        self.alphas=alphas
        self.N=2*p+l
        self.geometry=geometry # geometry paramters in which the mode exists
        self.paraxial=paraxial # paraxial or non-paraxial beam propagation

        self.z0 = np.sqrt( self.geometry.L*(self.geometry.Rm-self.geometry.L) )
        self.w0 = np.sqrt(self.z0*self.geometry.wavelength/PI)

    def angular_information(self):
        X,Y=self.geometry.get_2d_grid()
        return np.exp(1j*self.l*np.arctan2(Y,X))

    def field_profile(self,z=0):
        """
        Returns a normalized LG(p,l) mode, as a 2D field profile.
        """
        X,Y=self.geometry.get_2d_grid()

        wz=self.w0 * np.sqrt( 1 + (z/self.z0)**2 )
        scaling = np.sqrt(2)/wz # coordinate scaling
        rho = scaling*np.sqrt(X**2+Y**2) 

        if isinstance(self.p,int):
            normalization = (1)**self.p* np.sqrt(factorial(self.p)/(PI*factorial(self.p+self.l)) )# mode normalization
            field = (rho)**(self.l)*assoc_laguerre(rho**2,self.p,self.l) * np.exp(-rho**2/2) * normalization

        else:
            field=0
            for i,p in enumerate(self.p):
                normalization = (1)**p* np.sqrt(factorial(p)/(PI*factorial(p+self.l)) )# mode normalization
                field += (rho)**(self.l)*assoc_laguerre(rho**2,p,self.l) * np.exp(-rho**2/2)*normalization*self.alphas[i]

        return Field2D(field*scaling*self.angular_information(),self.geometry)

    def get_gouy(self,z):
        return -np.mod(np.arctan2(z,self.z0)*(self.N+1),PI)
    
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

        return fftpack.ifft2((psi_fourier*prop_vector))/self.angular_information()

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
# %%
def projection(e_mirr:Field1D,Psi:Mode):
    """
    Projects the field profile on the mirror surface onto a
    different p mode.
    """
    geometry=Psi.geometry
    psi=Psi.field_profile().cross_section()
    weight = geometry.xmax/geometry.grid_size*2*PI*abs(geometry.x)
    return np.sum(e_mirr*psi*weight)

def modal_decomposition(e_mirr:Field1D,Psi_0:Mode,tol=1e-4):
    """
    Calculates the projection of the electric field on the mirror
    to the LG{lp} basis. (for delta p>0 and <0 as long as p>=0)
    A tolerance of 1e-4 is the default, meaning that if the |alpha|^2<1e-4, the
    projection is too small to be taken into account.

    Returns the values of p, alpha and the projected mode.
    """
    p=Psi_0.p
    alpha_0 = projection(e_mirr,Psi_0)
    alpha_n=alpha_0
    ps = np.array([],dtype=int)
    alphas=np.array([])
    modes = np.array([])

    i=0
    # print("Delta p > 0")
    while abs(alpha_n)**2>tol:
        mode_plus = Mode(p+i,Psi_0.l,Psi_0.geometry,Psi_0.paraxial)
        modes=np.append(modes,mode_plus.field_profile().cross_section())
        alpha_n = projection(e_mirr,mode_plus)
        alphas=np.append(alphas,alpha_n)
        ps=np.append(ps,p+i)
        
        i+=1
    alpha_n=alpha_0
    i=0

    # print("Delta p < 0")
    while abs(alpha_n)**2>tol:
        i+=1
        if p-i>=0:
            mode_minus = Mode(p-i,Psi_0.l,Psi_0.geometry,Psi_0.paraxial)
            alpha_n = projection(e_mirr,mode_minus)
            alphas=np.insert(alphas,0,alpha_n)
            modes=np.insert(modes,0,mode_minus.field_profile().cross_section())
            ps=np.insert(ps,0,p-i)
        else:
            break
    return ps,alphas, modes*alphas
# %%
