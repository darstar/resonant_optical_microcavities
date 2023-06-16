"""
Simulations of optical microcavities - Fourier Propagation

Last edit: 13 Mar 2023 15:30

2D Fourier propagaion of Gaussian laser modes

Outdated version: does not contain routines or clear start of the main program
"""

#%%
import numpy as np
import matplotlib.pyplot as plt 
import time
import sys
from mpl_toolkits.mplot3d import Axes3D
PI = np.pi
#%%
# Setting paramters as in dimensions of the wavelength [micron]
wavelength = 1
Rm = 50*wavelength # radius of curvature
L= 10*wavelength # cavity length

#%%
# Grid discretization
p_wig = 1 # 0 for spherical mirror, 1 for parabloc mirror
grid_size = 2**8
x=np.linspace(-Rm*wavelength,Rm*wavelength,grid_size);y=np.copy(x);z=np.linspace(0,L,grid_size);r=np.sqrt(x**2+y**2)
dxy=max(x)/(grid_size)
X,Y=np.meshgrid(x,y)
R=X**2+Y**2

#%%
# Field labels
n=0
m = 0
l=abs(n+m)
p=min((n,m))  # also 1,2,3, later implementation
N = n+m
#%%
# Gaussian beam characteristics and geometry parameters
g = 1-L/Rm
w0 = (wavelength*L/PI*g/(1-g))**(0.5)
Gamma = 0.5 # tuning parameter for beam waist at z=0
k=2*PI/wavelength
w=Gamma*w0
z0 = PI*w0**2/wavelength;alpha=1/(k*z0)
wz=w * np.sqrt( 1 + (z/z0)**2 );gamma=wz/np.sqrt(2)

# Normalized Gaussian mode functions
def fundamental(X,Y,w0):
    """
    Input: 
        2D cartesian coordinates X and Y (meshgrid)
        Beam width at z=0
    --------------------------------------
    Output:
        Fundamental Gaussian mode
    """
    return 1j/(w0)*np.exp(-(X**2+Y**2)/w0**2)

from scipy.special import assoc_laguerre, eval_hermitenorm, factorial
def LG(n,m,X,Y,w0):
    """
    Input: 
        Mode numbers n and m
        2D cartesian coordinates X and Y (meshgrid)
        Beam width at z=0
    --------------------------------------
    Output:
        Normalized (l,p) Laguerre-Gauss mode
    """
    N=n+m
    l=np.abs(n-m)
    p=min((n,m))
    normalization = np.sqrt(2/PI) *factorial(p) / np.sqrt((factorial(n)*factorial(m)))
    scaling = np.sqrt(2)/w0
    r=X**2+Y**2
    r_norm = scaling**2*r
    theta=np.arctan2(Y,X)

    f = (r_norm)**(l/2)*assoc_laguerre(r_norm,p,l)

    return f*fundamental(X,Y,w0)*np.cos((theta*l))*normalization

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
    return normalization*eval_hermitenorm(n,scaling*X)*eval_hermitenorm(m,scaling*Y)*fundamental(X,Y,np.sqrt(2)*w0)
# %%
# 2D Fourier propagation
psi_0=LG(n,m,X,Y,w) # for now n=m=0, fundamental mode
def Fourier2D(psi_0,x,z,paraxial=False):
    """
    Input:
        psi_0: Slowly-varying field at z=0
        x: x coordinate, to determine kx values
        z: z coordinate to which to propagate
    ---------------------------------------------

    Output:
        Propagated, slowly-varying field at z.
    """
    k=2*PI
    psi_fourier = np.fft.fft2(psi_0)
    kx = np.fft.fftfreq(grid_size,max(x)/(grid_size))#There is a factor 2pi missing somewhere
    kxx,kyy=np.meshgrid(kx,kx)
    k=2*PI
    if paraxial:
        kz = ( k-(kxx**2+kyy**2)/(2*k))*2*PI
    else: 
        kz = ( k-(kxx**2+kyy**2)/(2*k) - (kxx**2+kyy**2)**2/(8*k**3) )*2*PI
    #(np.sqrt(k**2-kxx**2-kyy**2,dtype=complex))*2*PI
    prop_vector = np.exp(1j*(kz-k)*z)
    return np.fft.ifft2((psi_fourier*prop_vector))

from scipy import fft,fftpack
def Fourier2D_scipy(psi_0,x,z,paraxial=False):
    """
    Fourier function using scipy module, faster computation
    Input:
        psi_0: Slowly-varying field at z=0
        x: x coordinate, to determine kx values
        z: z coordinate to which to propagate
    ---------------------------------------------

    Output:
        Propagated, slowly-varying field at z.
    """
    k=2*PI
    psi_fourier = fft.fft2(psi_0)
    kx = fft.fftfreq(grid_size,max(x)/(grid_size))#There is a factor 2pi missing somewhere
    kxx,kyy=np.meshgrid(kx,kx)
    if paraxial:
        kz = ( k-(kxx**2+kyy**2)/(2*k))*2*PI
    else: 
        kz = ( k-(kxx**2+kyy**2)/(2*k) - (kxx**2+kyy**2)**2/(8*k**3) )*2*PI
    #(np.sqrt(k**2-kxx**2-kyy**2,dtype=complex))*2*PI
    prop_vector = np.exp(1j*(kz-k)*z)
    return fft.ifft2((psi_fourier*prop_vector))
def Fourier2D_scipy_pack(psi_0,x,z,paraxial=False):
    """
    Fourier function using scipy module, faster computation
    Input:
        psi_0: Slowly-varying field at z=0
        x: x coordinate, to determine kx values
        z: z coordinate to which to propagate
    ---------------------------------------------

    Output:
        Propagated, slowly-varying field at z.
    """
    k=2*PI
    psi_fourier = fftpack.fft2(psi_0)
    kx = fftpack.fftfreq(grid_size,max(x)/(grid_size))#There is a factor 2pi missing somewhere
    kxx,kyy=np.meshgrid(kx,kx)
    k=2*PI
    if paraxial:
        kz = ( k-(kxx**2+kyy**2)/(2*k))*2*PI
    else: 
        kz = ( k-(kxx**2+kyy**2)/(2*k) - (kxx**2+kyy**2)**2/(8*k**3) )*2*PI
    #(np.sqrt(k**2-kxx**2-kyy**2,dtype=complex))*2*PI
    prop_vector = np.exp(1j*(kz-k)*z)
    return fftpack.ifft2((psi_fourier*prop_vector))
psi_L=Fourier2D_scipy(psi_0,x,10,paraxial=False)
# %%
fig,ax=plt.subplots(1,2)
fig.suptitle(f"2D Angular spectrum propagation of {l}{p} mode\n intensity plot")
plt.xlabel('x')
plt.ylabel('y')
im1 = ax[0].imshow(np.abs(psi_0)**2,cmap='jet',origin='lower',extent=[-5,5,-5,5])
plt.colorbar(im1,ax=ax[0],orientation='horizontal')
ax[0].set_title(r"$\psi$(x,y,z=0)")
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
im2 = ax[1].imshow(np.abs(psi_L)**2,cmap='jet',origin='lower',extent=[-5,5,-5,5])
plt.colorbar(im2,ax=ax[1],orientation='horizontal')
ax[1].set_title(r"$\psi$(x,y,z=L)")
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
fig.tight_layout()

fig,ax=plt.subplots(1,2)
fig.suptitle(f"2D Angular spectrum propagation of {l}{p} mode \n real and imaginary part")
im1 = ax[0].imshow(psi_L.real,cmap='jet',origin='lower',extent=[-5,5,-5,5])
plt.colorbar(im1,ax=ax[0],orientation='horizontal')
ax[0].set_title(r"Re[$\psi$]")
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
im2 = ax[1].imshow(psi_L.imag,cmap='jet',origin='lower',extent=[-5,5,-5,5])
ax[1].set_title(r"Im[$\psi$]")
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
plt.colorbar(im2,ax=ax[1],orientation='horizontal')
fig.tight_layout()

print(f"Power of E at z=0: {np.sum(np.abs(psi_0.flatten())**2):.2f}")
print(f"Power of E at z=L: {np.sum(np.abs(psi_L.flatten())**2):.2f}")

#%%
# # 3D Plot
# fig = plt.figure(figsize=plt.figaspect(0.5))
# fig.suptitle(f"3D Vsiaulisation of {l}{p} mode propagation")
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.set_title("E(z,y,z=0)")
# im_E0 = ax.plot_surface(x, y, np.abs(E0)**2,cmap='jet',
#                        antialiased=False)

# fig.colorbar(im_E0, shrink=0.5)#, aspect=10)
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.set_title("E(z,y,z=L)")
# im_Ez=ax.plot_surface(x, y, np.abs(Ez)**2, cmap='jet')
# fig.colorbar(im_Ez, shrink=0.5)
# fig.tight_layout()
# plt.show()
# %%
# Interpolation around mirror surface
import scipy.interpolate as interpolate
zmin= L-Rm+np.sqrt(Rm**2-np.max(r)**2)
zm = (x**2+y**2)/(2*Rm) + (x**2+y**2)**2/(8*Rm**4)
z_planes = z[z>=zmin-L/grid_size][::4]
z_planes = np.append(z_planes,L)
psi_z=np.empty((grid_size,grid_size,len(z_planes)),dtype=complex)

#%%
# %%timeit # 21ms
# t_start = time.time()
# for i,plane in enumerate(z_planes):
#     psi_z[:,:,i]=Fourier2D(psi_0,x,plane)#[:,grid_size//2]
# t_end=time.time()
# print(f"FT numpy computation time: {t_end-t_start:.3f}s")
#%%
# %%timeit #18ms
# t_start = time.time()
# for i,plane in enumerate(z_planes):
#     psi_z[:,:,i]=Fourier2D_scipy(psi_0,x,plane)#[:,grid_size//2]
# t_end=time.time()
# print(f"FT scipy computation time: {t_end-t_start:.3f}s")
#%%
# %%timeit #18 ms
# t_start = time.time()
for i,plane in enumerate(z_planes):
    psi_z[:,:,i]=Fourier2D_scipy_pack(psi_0,x,plane,paraxial=True)#[:,grid_size//2]
# t_end=time.time()
# print(f"FT scipy fftpack computation time: {t_end-t_start:.3f}s")
#%%
# spline = interpolate.interp1d(z_planes,psi_z,axis=0,kind='cubic')
# psi_interp = spline(z_mirr)

psi_interp = interpolate.interpn((x,y,z_planes),psi_z,(x,y,L-zm)) #Continuous function of psi(x,y,z), calculated at points (x,y,L-zm)
print(f"Power of E at z=zm: {np.sum(np.abs(psi_interp)**2):.2f}")
ticks = [0,50,100,grid_size//2,150,200,250]
plt.figure()
plt.xlabel(r'$r$')
plt.ylabel(r"$|\psi(r,L-z_m)|^2$ ")
plt.plot(r,np.abs(psi_interp)**2)
# plt.gca().set_xticks(ticks)
# plt.gca().set_xticklabels((L-zm)[ticks].round(2))
plt.show()

plt.figure()
plt.xlabel(r'$L-z_m$')
plt.ylabel(r"Re[$\psi(r,L-z_m)$] ")
plt.plot(psi_interp.real)
plt.gca().set_xticks(ticks)
plt.gca().set_xticklabels((L-zm)[ticks].round(2))
plt.show()

plt.figure()
plt.xlabel(r'$L-z_m$')
plt.ylabel(r"Im[$\psi(r,L-z_m)$] ")
plt.plot(psi_interp.imag)
plt.gca().set_xticks(ticks)
plt.gca().set_xticklabels((L-zm)[ticks].round(2))
plt.show()

fig = plt.figure()
fig.suptitle(r"|$\psi(x,y,z_m)|^2$")
ax = plt.axes(projection="3d")
im = ax.scatter3D(x, y,L-zm, c=np.abs(psi_interp)**2, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(r'L-$z_m$')
plt.colorbar(im,orientation='horizontal',shrink=0.5)
plt.show()

#*%%
# Transverse and longitudinal components of the slowly varying field
#%%
def phase(psi):
    return np.arctan2(psi.imag,psi.real)

intensity = np.abs(psi_interp)**2
phi = phase(psi_interp)
phi_L = phase(psi_L)
plt.figure()
plt.title(r"Phase of $\psi(r,L-zm)$")
plt.xlabel(r'$L-z_m$')
plt.ylabel(r'$\phi [\pi]$')
# plt.gca().set_xticks(ticks)
# plt.gca().set_xticklabels((L-zm)[ticks].round(2))
plt.plot((phi)/PI,'.')

def avg_phase(psi):
    return np.average(phase(psi),weights=np.abs(psi)**2)
def rms_phase(psi):
    phi = phase(psi)
    return np.sqrt( np.average(phase(psi)**2,weights=np.abs(psi)**2) )

print(f"Average phase on mirror surface: {avg_phase(psi_interp)}")
print(f"RMS phase on mirror surface: {rms_phase(psi_interp)}")
print(f"On-axis phase of the slowly-varying field: {phase(psi_interp[grid_size//2])/PI}pi")

# %%
