#%%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import factorial,eval_genlaguerre
from scipy import fftpack
import scipy.interpolate as interpolate
from statsmodels.stats.weightstats import DescrStatsW
from scipy import integrate
PI=np.pi

class Geometry():
    """
    Geometry class, containing parameters that define the geometry of the optical microcavity.
    """
    def __init__(self,wavelength,L,Rm,pwig,grid_size,N):

        self.wavelength=wavelength
        self.k=2*PI/wavelength

        self.L=L
        self.Rm=Rm

        self.grid_size=grid_size
        self.pwig=pwig

        self.z0 = np.sqrt( self.L*(self.Rm-self.L) )
        self.w0 = np.sqrt(self.z0*self.wavelength/PI)

        self.xmax=8*np.sqrt((N+1))*self.w0
        self.x = np.linspace(-self.xmax,self.xmax,self.grid_size,dtype=np.double)

    def get_2d_grid(self):
        X,Y=np.meshgrid(self.x,self.x)
        return X,Y

    def mirror_coords(self):  
        """
        Returns the mirror coordinates zm=z(x,y).
        """      
        return self.x**2/(2*self.Rm) + self.x**4/(8*self.Rm**3)*(1-self.pwig)
        #pwig is wrong  zmirr - zsphere = -pwig r^3/8Rm^3, sphere
    def mirror_planes(self,n):
        """
        Returns n planes around the mirror surface, used in the interpolation routine in the 'Mode' class.
        """
        zmin = self.L-self.Rm+np.sqrt(self.Rm**2-self.xmax**2/2,dtype=np.double)
        return np.linspace(zmin,self.L,n,dtype=np.double)
    
class Field():

    def __init__(self,profile,geometry:Geometry):
        self.geometry=geometry
        self.profile=profile

        self.abs=np.abs(self.profile)
        self.intensity=self.abs**2
 
        self.real=self.profile.real
        self.imag=self.profile.imag

        self.phase=np.angle(self.profile)#np.arctan2(self.imag,self.real)#p.angle(self.profile)
        self.phase_rectified = np.angle(self.profile * np.sign(self.real) )
    
    def __mul__(self, other):
        """
        Operator overload, making a 'Field' object able to
        multiple with a numpy array, a numer or another Field object.
        """

        if isinstance(other, self.__class__):
            return Field(self.profile*other.profile,self.geometry)
        elif isinstance(other, complex) or isinstance(self, np.ndarray):
            return Field(self.profile*other,self.geometry)
    
class Field1D(Field):

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
        x=self.geometry.x
        weight = PI*abs(x)#*2*geometry.xmax/geometry.grid_size
        arg = interpolate.CubicSpline(x,abs(self.profile)**2*weight)
        return arg.integrate(-self.geometry.xmax,self.geometry.xmax)
        # spacing=self.geometry.xmax/self.geometry.grid_size
        # weight = abs(self.geometry.x)*2*PI*spacing
        # return np.sum(self.intensity*weight,dtype=np.double)
    
    def phase_moments(self):
        """
        Returns the mean and RMS phase of the field
        """

        weights= self.intensity*abs(self.geometry.x)*PI
        moments = DescrStatsW(self.phase_rectified,weights)
        return moments.mean, np.sqrt(moments.var)
    
    def phase_shift(self):
        """
        Shifts the field over the average phase
        """
        return self*np.exp(-1j*self.phase_moments()[0])
    
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
        return np.sum(self.intensity.flatten()*spacing**2,dtype=np.double)

    def cross_section(self):
        """
        Returns a 1D field, containing the profile cross-sected at y=0.
        """
        return Field1D(self.profile[:,self.geometry.grid_size//2],self.geometry)
    
class Mode():
    """Class containing paraxial eigenmodes of the cavity (LG basis)."""

    def __init__(self,p,l,N,geometry:Geometry,paraxial):
        self.p=p
        self.l=l
        self.N=N

        self.geometry=geometry # geometry paramters in which the mode exists
        self.paraxial=paraxial # paraxial or non-paraxial beam propagation

        self.z0 = np.sqrt( self.geometry.L*(self.geometry.Rm-self.geometry.L),dtype=np.double )
        self.w0 = np.sqrt(self.z0*self.geometry.wavelength/PI,dtype=np.double)
        self.wL=self.w0*np.sqrt(1+(self.geometry.L/self.z0)**2)
    
    def angular_information(self):
        X,Y=self.geometry.get_2d_grid()

        return np.exp(1j*(self.l)*np.arctan2(Y,X),dtype=complex)
        
    def field_profile(self,z=0):
        """
        Returns a normalized LG(p,l) mode, as a 2D field profile.
        """
        X,Y=self.geometry.get_2d_grid()

        wz=self.w0 * np.sqrt( 1 + (z/self.z0)**2 )
        scaling = np.sqrt(2)/wz # coordinate scaling
        rho = scaling*np.sqrt(X**2+Y**2) 

        normalization = (1)**self.p* np.sqrt(factorial(self.p)/(PI*factorial(self.p+self.l)) )# mode normalization
        field = (rho)**(self.l)*eval_genlaguerre(self.p,self.l,rho**2) * np.exp(-rho**2/2) * normalization

        # # Possible extension: compose a field profile from the modal decomposition
        # elif isinstance(self.p,np.ndarray):
        #     field=0
        #     rows = self.alphas.shape[0]
        #     cols = self.alphas.shape[1]
        #     for i in range(rows):
        #         for j in range(cols):
        #             normalization = (1)**self.p[i]* np.sqrt(factorial(self.p[i])/(PI*factorial(self.p[i]+self.l)) )# mode normalization
        #             field += (rho)**(self.l)*assoc_laguerre(rho**2,self.p[i],self.l) * np.exp(-rho**2/2)*normalization*self.alphas[i][j]
        
        return Field2D(field*scaling*self.angular_information(),self.geometry)

    def get_gouy(self,z):
        return -np.arctan2(z,self.z0)*(self.N+1)
    
    def FFT2D(self,z):
        kbar=self.geometry.k/(2*PI)
        psi_fourier = fftpack.fft2(self.field_profile().profile)
        kbar_x = fftpack.fftfreq(self.geometry.grid_size,2*self.geometry.xmax/self.geometry.grid_size)
        kxx,kyy=np.meshgrid(kbar_x,kbar_x)

        if self.paraxial: # Paraxial propagation, first  order Taylor expansion
            kbar_z = kbar-(kxx**2+kyy**2)/(2*kbar)
        else:  # Non paraxial propagation, all higher order terms are taken into account by the sqrt
            kbar_z = np.sqrt(kbar**2-kxx**2-kyy**2,dtype=complex)

        prop_vector = np.exp(1j*(kbar_z-kbar)*2*PI*z,dtype=complex)

        return fftpack.ifft2((psi_fourier*prop_vector))/self.angular_information()

    def propagate_mirr(self,n=4):
        """Returns the corrected slowly-varying field at the mirror surface"""
        z_planes = self.geometry.mirror_planes(n)
        psi_z=np.empty((self.geometry.grid_size,len(z_planes)),dtype=complex)

        for i,plane in enumerate(z_planes):
            psi_z[:,i]=self.FFT2D(plane)[:,self.geometry.grid_size//2] # cross-section of the propagated field

        interp_fcn = interpolate.interp1d(z_planes,psi_z,kind='cubic',fill_value="extrapolate")
        psi_e = np.diag( interp_fcn(self.geometry.L-self.geometry.mirror_coords()) ) * np.exp(-1j*self.geometry.k*(self.geometry.mirror_coords()))

        return Field1D( psi_e, self.geometry )
        
    def propagate_plane(self):
        """Returns the electric field at the z=L plane"""
        psi_e = self.FFT2D(self.geometry.L)[:,self.geometry.grid_size//2] * np.exp(-1j*self.geometry.k*(self.geometry.mirror_coords()))
        return Field1D(psi_e,self.geometry)

# def complex_quadrature(func, a, b):
#     def real_func(x):
#         return (func(x)).real
#     def imag_func(x):
#         return (func(x)).imag
    
#     real_integral = integrate.quad(real_func, a, b)
#     imag_integral = integrate.quad(imag_func, a, b)

#     return real_integral[0]+1j*imag_integral[0]

def projection(e_mirr:Field1D,Psi:Mode,p0):
    """
    Projects the field profile on the mirror surface onto a
    different p mode.
    """

    geometry=Psi.geometry
    psi=Psi.field_profile(geometry.L).cross_section()
    if Psi.p%2!=0:
        x=geometry.x*np.sqrt(2)/Psi.wL
    else:
        x=geometry.x
    weight = PI*abs(x)#*2*geometry.xmax/geometry.grid_size
    arg = interpolate.CubicSpline(x,e_mirr.profile*psi.profile*weight)
    return arg.integrate(-x[-1],x[-1])

def modal_decomposition(e_mirr:Field1D,Psi_0:Mode):
    """
    Calculates the projection of the electric field on the mirror
    to the LG{lp} basis. (for delta p>0 and <0 as long as p>=0)

    Returns the values of p, alpha and the projected mode.
    """
    geometry=e_mirr.geometry
    e_mirr=e_mirr.phase_shift()
    p=Psi_0.p
    alpha_0 = projection(e_mirr,Psi_0,Psi_0.p)
    alpha_n=alpha_0
    ps = np.array([],dtype=int)
    alphas=np.array([])
    modes = np.array([])

    i=0
    # print("Delta p > 0")
    while i<=2:
        mode_plus = Mode(p+i,Psi_0.l,2*Psi_0.N,Psi_0.geometry,Psi_0.paraxial)
        modes=np.append(modes,mode_plus.field_profile(geometry.L).cross_section())
        alpha_n = projection(e_mirr,mode_plus,Psi_0.p)
        alphas=np.append(alphas,alpha_n)
        ps=np.append(ps,p+i)
        i+=1
    alpha_n=alpha_0
    i=0

    # print("Delta p < 0")
    while i<=2:
        i+=1
        if p-i>=0:
            mode_minus = Mode(p-i,Psi_0.l,Psi_0.N/2,Psi_0.geometry,Psi_0.paraxial)
            alpha_n = projection(e_mirr,mode_minus,Psi_0.p)
            alphas=np.insert(alphas,0,alpha_n)
            modes=np.insert(modes,0,mode_minus.field_profile(geometry.L).cross_section())
            ps=np.insert(ps,0,p-i)
        else:
            break
    return ps,alphas, modes*alphas

def expected_coupling(p,l,strength):
    """Returns the expected modal coupling amplitude for a (p,l) mode and
    relative strengths"""

    min2=np.sqrt((p-1)*p)*np.sqrt((p+l)*(p+l+1))
    min1=-4*p*np.sqrt(p*(p+l))
    plus1=-4*(p+1)*np.sqrt((p+1)*(p+l+1))
    plus2=np.sqrt((p+1)*(p+2))*np.sqrt((p+l+1)*(p+l+2))
    if isinstance(strength,float):
        return np.array([min2,min1,plus1,plus2])*strength
    else:
        coeff=np.array([min2,min1,plus1,plus2])
        result=coeff
        for val in strength:
            result = np.vstack((result,coeff*val))
        return result[1:]