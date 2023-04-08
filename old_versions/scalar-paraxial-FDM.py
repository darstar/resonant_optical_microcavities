"""

Simulations of optical microcavities - first draft

Last edit: 23 Feb 2023 10:00

Solving the slowly varying, helmholz equation for the 00 Gauss mode by using the Finite Difference Method

--> some kind of instability around r=0, which does not give the correct answers

"""
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
L= 20*wavelength # cavity length
#%%
# Grid discretization

grid_size = 2**8
x=np.linspace(0,10*wavelength,grid_size+1);y=np.copy(x);z=np.linspace(0,L,grid_size+1);r=np.sqrt(x**2+y**2)
X,Y=np.meshgrid(x,y)
R=np.sqrt(X**2+Y**2);np.arctan2(y,x)
z_mirr = np.sqrt(Rm**2-y**2)+L-Rm
#%%
# Field labels
l = 0
p = 0 # also 1,2,3, later implementation
N = 2*p + l
#%%

# Beam profile
k=2*np.pi/wavelength
w0=5*wavelength
z0 = np.pi*w0**2/wavelength;alpha=1/(k*z0);gouy = np.arctan2(z,z0);qz=z-1j*z0
Rz=w0*np.sqrt(1+(z0/z)**2)
wz=w0 * np.sqrt( 1 + (z/z0)**2 );gamma=wz/np.sqrt(2)
R_norm=R/gamma
#%%

# Modes
from scipy.special import genlaguerre
radial_profile=np.sqrt(1/np.pi)*1/wz*np.exp(-R**2/(wz**2))*np.exp(-1j*(R**2/(2*Rz)+gouy))
# radial_profile = 1j/wz*R_norm**l * genlaguerre(p,l)(R_norm**2) * np.exp(-R_norm**2/2)#*np.exp(1j*gouy)
# LG_mode = radial_profile * np.exp(1j*l*theta) * np.exp( -1j*(N+1)*gouy)
plt.figure()
plt.title(f"{l}{p} mode at z=0")
plt.imshow(np.abs(radial_profile)**2,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],cmap='jet',origin='lower')
plt.colorbar()

# #%%
plt.figure()
plt.title(f"Sketch of the optical cavity for \n L={L},Rm={Rm}")
plt.xlabel("z");plt.ylabel("y")
plt.xlim(-0.1,L+0.1)
plt.gca().set_xticks([0,np.sqrt(2)*w0,L]);plt.gca().set_xticklabels(["0",r"$\sqrt{2}w_0$","L"])
plt.gca().set_yticks([-5*wavelength,0,5*wavelength]);plt.gca().set_yticklabels([r"-5$\lambda$","0","5$\lambda$"])
plt.plot(z_mirr,y,color='r',marker='.',markersize=1,ls='none')
plt.vlines(0,ymin=np.min(x),ymax=np.max(x),color='r')
# plt.savefig(f"Figures/cavity_sketch_L{L}_R{Rm}")
plt.show()

# %%
# Solving the modified, slowly varying Helmholz equation
import scipy.sparse as sp
import scipy.sparse.linalg as la

# Step size
hr=np.max(R_norm)/grid_size
hz=L/grid_size
nums=np.arange(2,grid_size)
eta=2*k*hr**2/hz

if eta<1:
    print(rf"$\nu$ = {eta} < 1, we can continue with the computations")
else:
    print(rf"$\nu$ = {eta} > 1, choose different grid_size")
    sys.exit(0)
# Diagonal matrices, FDM Disrectization

diagonal_main=np.full(grid_size-1,-2)
diagonal_upper=np.full(grid_size-2,(1+1/nums))
diagonal_lower=np.full(grid_size-2,(1-1/nums))

diagonal_upper[0]=2

D=sp.diags([diagonal_upper,diagonal_main,diagonal_lower],offsets=[1,0,-1],format="csc") # Matrix first derivative operator in r
Dinv=1j*eta*la.inv(D)

psi_propagated = np.zeros((grid_size+1,len(z)),dtype=complex)
psi_propagated[:,0]=radial_profile[0]
psi_propagated[:,1]=psi_propagated[:,0]
psi0=psi_propagated[:,0][1:-1]
I=sp.eye(grid_size-1,format="csc")

forward_step=la.inv(I-0.5*Dinv)@(I-0.5*Dinv) # based on 'Numerical recipes', unitary matrix

# Backward Euler iterations for wave porpagation
start_time=time.time()
for n in range(1,len(z)):
    psi_transverse=forward_step@psi0
    psi0=psi_transverse
    psi_transverse=np.insert(psi_transverse,obj=0,values=psi_transverse[0])
    psi_transverse=np.insert(psi_transverse,obj=-1,values=0)

    psi_propagated[:,n]=psi_transverse
end_time=time.time()

print(f"Computation time for {grid_size}x{grid_size} grid: {(end_time-start_time):.4f}s")
newline='\n'
plt.figure(dpi=400)
plt.title(rf"Propagation of {l}{p} mode by solving {newline} the paraxial Helmholz equation, {newline} Finite Differnce method {newline} $\alpha$={alpha:.4f},$\eta$={eta:.4f} {newline} time={(end_time-start_time):.4f}s")
plt.xlabel("z")
plt.ylabel("r")
# plt.ylim(0,w0)
# plt.gca().set_yticks([0,w0,L])
# plt.gca().set_yticklabels(["0",r"$w_0$","L"])
# plt.gca().set_xticks([0,L])
# plt.gca().set_xticklabels(["0","w0","L"])
plt.plot(z,wz,color='white')
plt.imshow(np.abs(psi_propagated*np.exp(1j*k*z))**2,cmap='jet',extent=[0,L,np.min(r),np.max(r)],origin='lower')
plt.colorbar()
# %%
plt.figure(dpi=400)
plt.xlabel('r')
plt.ylabel(r'Intensity $|\psi|^2$')
plt.plot(r,np.abs(psi_propagated[:,0])**2,'.')
plt.plot(r,np.abs(psi_propagated[:,2])**2,'.')
plt.plot(r,np.abs(psi_propagated[:,-1])**2,'.')
plt.show()

# plt.plot(z,wz)
#%%
