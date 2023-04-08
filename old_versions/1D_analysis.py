"""
Interpolation based on 1D analysis of the radial profile

Last edit: 21 Mar 2023 10:09

"""
#%%
from FourierPropagation_routines1D import *
PI=np.pi

# %%
# Setting paramters as in dimensions of the wavelength [micron]
wavelength = 1
Rm = 50*wavelength # radius of curvature
L= 10*wavelength # cavity length
n=0
m = 0
l=abs(n+m)
p=min((n,m))
N = n+m
k=2*PI/wavelength # Wave vector
z0 = np.sqrt(L*(Rm-L))
w0 = np.sqrt(z0*wavelength/PI)
# Grid discretization
p_wig = 1 # 0 for spherical mirror, 1 for parabloc mirror
grid_size = 2**8
xmax=10*wavelength
x=np.linspace(-xmax,xmax,grid_size);y=np.copy(x) # Cartesian coordinates
r=np.sqrt(x**2+y**2);theta=np.arctan2(y,x) # Cylindrical cooridnates
z=np.linspace(0,L,grid_size) # z range from 0 to cavity length

#%%
# Main computations: using subroutines from 'FourierPropagation_routines'
# to calculate the field at the mirror surface
psi_0=LG(x,y,p,l,w0)#fundamental(X,Y,w0) # field at z=0
psi_L=Fourier1D_scipy_pack(psi_0,r,L) 

plt.plot(r,abs(psi_0)**2)
plt.plot(r,abs(psi_L)**2)
#%%
# Interpolation around mirror surface
zmin= L-Rm+np.sqrt(Rm**2-2*xmax**2) #determining a minimum z value from which to interpolate, based on the geometry of tha cavity

zm = Rm - np.sqrt(Rm**2-r**2) # mirror surface coordinates
z_planes = np.linspace(zmin,L,4) #choose 4 z planes on the defined grid
xm=np.sqrt(Rm**2-(Rm-L+z_planes)**2)/2
psi_z=np.empty((grid_size,len(z_planes)),dtype=complex)

for i,plane in enumerate(z_planes):
    # Propagation of the initial field to 4 planes around the mirror surface
    psi_z[:,i]=Fourier1D_scipy_pack(psi_0,x,plane,paraxial=True)

# Interpolation: creating a continuous function around the mirror surface,
# and evaluating psi(x,y,L-zm) at the points calculated previously
psi_interp_1d = interpolate.interpn((x,z_planes),psi_z,(x,L-zm))

# interp_1d=interpolate.interp1d(z_planes,psi_z,kind='cubic')
# psi_interp_1d=interp_1d(L-zm)
# psi_interp_1d=psi_interp_1d.T[grid_size//2][:]

phase = calc_phase(psi_interp_1d)
phase_avg = avg_phase(psi_interp_1d)
phase_mean=np.mean(phase)
phase_rms = rms_phase(psi_interp_1d)
phase_on_axis = calc_phase(psi_interp_1d)[grid_size//2]

psi_phase_shifted = phase_shift(psi_interp_1d)

wL=w0 * np.sqrt( 1 + (L/z0)**2 );gouy=np.arctan2(L,z0)

psi_expected=LG(x,y,n,m,wL)*gouy_shift(L,z0,N)
psi_expected = phase_shift(psi_expected)

#%%
plt.figure()
plt.title("Fundamental mode propagation to 4 planes around mirror,\n \
          compared to the paraxial expectations")
plt.xlabel(r'$r \;[\mu m]$')
plt.ylabel(r"$|[\psi]|$")
for i in range(4):
    plt.plot(r,abs(psi_z.T[i]),'.',label=f'Field at z={z_planes[i]:.2f}')
plt.plot(r,abs(psi_expected),color='k',ls='--',label='Paraxial expectation')
plt.legend()
plt.show()

plt.figure()
plt.title(f"Real part of the phase shifted field on the mirror surface \n \
          Average phase = {phase_avg:.2f}\n \
          RMS phase = {phase_rms:.2f}\n \
          On-axis phase = {phase_on_axis:.2f}\n \
          alpha = {1/(k*z0):.4f}, L={L}, Rm={Rm}")
plt.xlabel(r'$r \;[\mu m]$')
plt.ylabel(r"$Re[\psi]$")
plt.plot(r,psi_phase_shifted.real,'.',label='Interpolated (phase shifted) values')
plt.plot(r,psi_0.imag,ls='--',color='r',label='Field at z=0')
plt.plot(r,psi_expected.real,'k--',label='Expected paraxial result')
plt.legend()
plt.show()

# %%