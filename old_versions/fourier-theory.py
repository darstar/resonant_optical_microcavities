#%%
import numpy as np
import matplotlib.pyplot as plt 
import time
import sys
#%%
# Implementation of the scalar, paraxial solution to the Helmholz-like equation
#%%
wavelength = 1 # all parameters set to wavelength unit
Rm = 20*wavelength # radius of curvature
L= 50*wavelength # cavity length
#%%
# Grid discretization

grid_size = 2**8
x=np.linspace(-5*wavelength,5*wavelength,grid_size+1);y=np.copy(x);z=np.linspace(0,L,grid_size+1);r=np.sqrt(x**2+y**2)
dxy=max(x)/(grid_size+1)
X,Y=np.meshgrid(x,y)

# Beam profile
k=2*np.pi/wavelength
w0=2*wavelength
z0 = np.pi*w0**2/wavelength;alpha=1/(k*z0);gouy = np.arctan2(z,z0);qz=z-1j*z0
wz=w0 * np.sqrt( 1 + (z/z0)**2 );gamma=wz/np.sqrt(2)
#%%

# Modes
from scipy.special import genlaguerre
radial_profile=1j*np.exp(-(X**2+Y**2)/w0**2)

kx = np.fft.fftshift(np.fft.fftfreq(grid_size+1,d=dxy))
ky = np.fft.fftshift(np.fft.fftfreq(grid_size+1,d=dxy))

kxx,kyy=np.meshgrid(kx,ky)

gauss0=np.exp(-X**2/w0**2)

plt.figure()
# plt.plot(x,np.abs(gauss0),label='z=0')
for i,z_plane in enumerate(z):
    if i%50==0 and i!=0:
        fourier_width= 1/(4*w0**2)+z_plane/(2*k)+z_plane/(8*k**3)
        plt.plot(x,np.sqrt(np.pi/fourier_width)*np.exp(-x**2/(4*fourier_width)),label=f'z={z_plane:.2f}')
plt.legend()
plt.show()

# %%
psi_propagated=np.exp(-(X**2+Y**2)/(4*fourier_width))

# %%

plt.figure()
plt.imshow(psi_propagated)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(radial_profile))
plt.colorbar()
# %%
from matplotlib.colors import Normalize
normalizer=Normalize(0,np.max(np.abs(radial_profile)**2))
fig,ax=plt.subplots(nrows=1,ncols=2)
fig.suptitle("Theoretical prediction for \n propagation of TEM00 mode")
im1 = ax[0].imshow(np.abs(radial_profile)**2,cmap='jet',origin='lower',extent=[0,5,0,5])#,norm=normalizer)
ax[0].set_title("E(x,y,0)")
im2 = ax[1].imshow(np.abs(psi_propagated)**2,cmap='jet',origin='lower',extent=[0,5,0,5])#,norm=normalizer)
ax[1].set_title("E(x,y,L)")
# fig.colorbar(im1)
# %%
