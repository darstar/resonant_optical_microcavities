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
on axis phase calculation?? 
to implement: operator overload '*' for 'Field' and np array
higher p modes??

if parax and calculated are very different --> maybe a different profile??
"""
class Geometry():
    def __init__(self,wavelength,L,Rm,pwig,grid_size):
        self.wavelength=wavelength
        self.k=2*PI*wavelength
        self.L=L*wavelength
        self.Rm=Rm*wavelength
        self.grid_size=grid_size
        self.pwig=pwig

        self.xmax=xmax=np.sqrt(self.Rm*self.L)/2
        self.x = np.linspace(-xmax,xmax,self.grid_size)

    def get_2d_grid(self):
        X,Y=np.meshgrid(self.x,self.x)
        return X,Y

    def mirror_coords(self):        
        return self.L-self.x**2/(2*self.Rm) - self.x**4/(8*self.Rm**3)*(1-self.pwig)
    
    def mirror_planes(self):
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
    
class Field1D(Field):
    def __init__(self,profile,geometry:Geometry):
        self.geometry=geometry
        self.profile=profile
        self.abs=np.abs(self.profile)
        self.intensity=self.abs**2
        self.phase=np.angle(self.profile)
        self.real=self.profile.real
        self.imag=self.profile.imag
        self.phase_on_axis=self.phase[self.geometry.grid_size//2]

    def power(self):
        spacing=self.geometry.xmax/self.geometry.grid_size
        weight = abs(self.geometry.x)*2*PI*spacing
        return np.sum(self.intensity*weight)
    
    def phase_moments(self):
        weights=self.intensity*abs(self.geometry.x)*PI
        moments = DescrStatsW(self.phase,weights)
        return moments.mean, abs(moments.mean**2 - moments.var)
    
    def phase_shift(self):
        return Field1D(self.profile*np.exp(-1j*self.phase_moments()[0]),self.geometry)
    
    def phase_plate(self):
        return Field1D(self.profile*np.exp(1j*(L-self.geometry.k*self.geometry.mirror_coords())),self.geometry)

class Field2D(Field):

    def power(self):
        spacing = 2*self.geometry.xmax/self.geometry.grid_size
        return np.sum(self.intensity.flatten()*spacing**2)

    # def phase_moments(self):
    #     weights=self.intensity*abs(self.geometry.get_2d_grid()[0])*PI
    #     moments = DescrStatsW(self.phase,weights)
    #     return moments.mean, np.sqrt(moments.var)
    
    def cross_section(self):
        return Field1D(self.profile[self.geometry.grid_size//2],self.geometry)
    
    def plot_intensity(self,**kwargs):
        plt.xlabel(r"x [$\mu m$]");plt.ylabel(r"y [$\mu m$]")
        plt.imshow(self.intensity,cmap='jet',origin='lower', \
                   extent=[-self.geometry.xmax,self.geometry.xmax,-self.geometry.xmax,self.geometry.xmax],**kwargs)
        plt.colorbar()

class Mode():
    def __init__(self,p,l,geometry:Geometry,paraxial):
        self.p=p
        self.l=l
        self.N=2*p+l
        self.geometry=geometry
        self.paraxial=paraxial
        self.z0 = np.sqrt( self.geometry.L*(self.geometry.Rm-self.geometry.L) )
        self.w0 = np.sqrt(self.z0*self.geometry.wavelength/PI)

    def field_profile(self,z=0):
        X,Y=self.geometry.get_2d_grid()
        normalization = (-1)**self.p* np.sqrt(factorial(self.p)/(PI*factorial(self.p+self.l)) )

        wz=self.w0 * np.sqrt( 1 + (z/self.z0)**2 )
        scaling = np.sqrt(2)/wz

        rho = scaling*np.sqrt(X**2+Y**2)
        theta=np.arctan2(Y,X)

        f = (rho)**(self.l)*assoc_laguerre(rho**2,self.p,self.l) * np.exp(-rho**2/2)
        return Field2D(f*normalization*scaling,self.geometry)

    def get_gouy(self,z):
        return np.arctan2(z,self.z0)*(self.N+1)
    
    def angular_spectrum(self,z):
        kbar=self.geometry.k/(2*PI)
        psi_fourier = fftpack.fft2(self.field_profile().profile)
        kbar_x = fftpack.fftfreq(self.geometry.grid_size,2*self.geometry.xmax/self.geometry.grid_size) # domain goed from negative to positive, so 2*maximal value !!
        kxx,kyy=np.meshgrid(kbar_x,kbar_x)

        if self.paraxial: # Paraxial propagation, first  order Taylor expansion
            kbar_z = kbar-(kxx**2+kyy**2)/(2*kbar)
        else:  # Non paraxial propagation, all higher order terms are taken into account by the sqrt
            kbar_z = np.sqrt(kbar**2-kxx**2-kyy**2,dtype=complex)

        prop_vector = np.exp(1j*(kbar_z-kbar)*2*PI*z)
        return fftpack.ifft2((psi_fourier*prop_vector))

    def propagate_mirr(self):
        z_planes = self.geometry.mirror_planes()
        psi_z=np.empty((self.geometry.grid_size,len(z_planes)),dtype=complex)
        for i,plane in enumerate(z_planes):
            psi_z[:,i]=self.angular_spectrum(plane)[:,self.geometry.grid_size//2]
        interp_fcn = interpolate.interp1d(z_planes,psi_z,kind='cubic')
        electric_field = np.diag( interp_fcn(self.geometry.mirror_coords()) ) * np.exp(1j*self.geometry.k*(self.geometry.mirror_coords()))
        return Field1D( electric_field, self.geometry )
    
    def propagate_plane(self):
        electric_field = self.angular_spectrum(self.geometry.L)[self.geometry.grid_size//2]*np.exp(1j*self.geometry.k*self.geometry.L)
        return Field1D(electric_field,self.geometry)

#%%

def plot_mirr(e_mirr:Field1D,e_0:Mode,geometry):
    phase_avg,phase_rms=e_mirr.phase_moments()
    e_phase_shifted = e_mirr.phase_shift()
    e_expected = e_0.field_profile(geometry.L)
    print(e_phase_shifted.phase[geometry.grid_size//2])
    if e_0.paraxial:
        paraxial='paraxial'
    else:
        paraxial='non-paraxial'
    fig,axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(wspace=0.3)
    plt.xlabel(r"x [$\mu m$]")
    axs[0].set_title(f"Phase shifted electric field on the mirror surface \n ({paraxial} propagation of {e_0.l}{e_0.p} mode ) \n \
        Average phase = {phase_avg:.4f}, RMS phase = {phase_rms:.4f}\n \
        On-axis phase = {e_mirr.phase_on_axis:.4f}, Paraxial phase = {-e_0.get_gouy(geometry.L):.4f}\n \
        L/(Rm-L)={geometry.L/(geometry.Rm-geometry.L)}, L={geometry.L}, Rm={geometry.Rm}",fontsize=10)
    
    axs[0].set_ylabel(r"$E [V \mu m^{-1}]$")
    axs[0].plot(abs(geometry.x),(e_phase_shifted.real),'.',label=r'Re[$E$]')
    axs[0].plot(abs(geometry.x),(e_phase_shifted.imag),':',label=r'Im[$E$]')
    # axs[0].plot(abs(geometry.x),(100*e_phase_shifted.imag),color='orange',label=r'100*Im[$E$]')
    axs[0].plot(abs(geometry.x),e_phase_shifted.abs,'.',markersize=2,label=r'$|E|$')
    
    axs[0].plot(abs(geometry.x),e_expected.cross_section().abs,'k--',label='Expected paraxial result',alpha=0.75)
    axs[0].legend(fontsize=8)

    axs[1].set_title(f"Phase of phase shifted electric field",fontsize=10)
    axs[1].set_ylabel(r'Arg $E$ [rad]')
    axs[1].plot(abs(geometry.x),e_phase_shifted.phase,'r')
    plt.legend()
    plt.show()

def plot_plane(e_plane:Field1D,e_0:Mode,geometry):
    # e_corrected = e_plane.phase_plate()
    e_phase = e_plane.phase + geometry.k*(geometry.mirror_coords()-L)
    mask=(geometry.x>0)&(geometry.x<7)
    if e_0.paraxial:
        paraxial='paraxial'
    else:
        paraxial='non-paraxial'
    plt.figure()
    plt.title(f"Corrected phase of electric field at z=L plane \n ({paraxial} propagation)\n"+ \
              r"$Arg[E]-k\cdot z_m$")
    plt.xlabel(r'$x \;[\mu m]$')
    plt.ylabel(r"$Arg \;[rad]$")
    plt.axhline(-e_0.get_gouy(geometry.L),xmin=0,xmax=1,ls='--',color='r',label=f'Paraxial phase {-e_0.get_gouy(geometry.L):.4f}')
    plt.axhline(e_plane.phase_on_axis,xmin=0,xmax=1,ls='--',color='k',label=rf'On-axis phase: {e_plane.phase_on_axis:.4f}')
    plt.plot(geometry.x[mask],e_phase[mask],label=r'$\psi(z=L)\cdot \exp(ikz_m)$')
    plt.legend()
    plt.show()

#%%
wavelength=1
L=10
Rm=50
grid_size=2**8-1
pwig = 1
geometry = Geometry(wavelength,L,Rm,pwig,grid_size)

l=0
p=0
paraxial=True

Psi_0=Mode(p,l,geometry,paraxial)
psi_0=Psi_0.field_profile().cross_section()
psi_expected=Psi_0.field_profile(L).cross_section()

# %%

psi_mirr = Psi_0.propagate_mirr()
psi_plane=Psi_0.propagate_plane()
# %%
plot_mirr(psi_mirr,Psi_0,geometry)
plot_plane(psi_plane,Psi_0,geometry)
# %%