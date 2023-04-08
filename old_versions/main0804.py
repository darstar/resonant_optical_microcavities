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
from FourierPropagation_routines import *
PI=np.pi
# %%
# Setting paramters as in dimensions of the wavelength [micron]
wavelength = 1
Rm = 50*wavelength # radius of curvature
L= 10*wavelength # cavity length
k=2*PI/wavelength # Wave vector
z0 = np.sqrt(L*(Rm-L))
w0 = np.sqrt(z0*wavelength/PI)
p_wig = 1 # 0 for spherical mirror, 1 for parabloc mirror
l=0;p=0;N=2*p+l

# Grid discretization
grid_size = 2**8-1
xmax=np.sqrt(L*Rm)/2*wavelength
x=np.linspace(-xmax,xmax,grid_size);y=np.copy(x) # 1D arrays of Cartesian coordinates
z=np.linspace(0,L,grid_size) # z range from 0 to cavity length
X,Y=np.meshgrid(x,y) # 2D arrays of Cartesian coordinates
zm = Rm - np.sqrt(Rm**2-x**2) # 1D array containing the z coordinates of the mirror
                                # cross section at y=0
#%%
# # Main computations 
# Choice between paraxial/nonparaxial and mirror/plane propagation
paraxial=False
mirror=True

if paraxial:
    prop='paraxial'
else:
    prop='non-paraxial'

Psi_0=LG(p,l,X,Y,w0) # 2D field at z=0
print(f"Power of LG{l}{p} mode: {power(Psi_0,2*max(x)/grid_size)}") # power check

wL=w0 * np.sqrt( 1 + (L/z0)**2 );gouy=np.arctan2(L,z0)
# Computations, depending on choice
if mirror:
    # Propagation to curved mirror
    zmin= L-Rm+np.sqrt(Rm**2-2*xmax**2) #determining a minimum z value from which to interpolate, based on the geometry of tha cavity
    z_planes = np.linspace(zmin,L,4) #choose 4 z planes on the defined grid
    e_interp = propagate(Psi_0,x,L-zm,paraxial=paraxial,z_planes=z_planes)*np.exp(-1j*k*L) #propagation function, giving back the electric field
    phase_avg,phase_rms=phase_moments(e_interp,abs(x))
    e_phase_shifted=phase_shift(e_interp,abs(x)) #subtracting the average phase

    fig,axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(wspace=0.3)
    plt.xlabel(r"x [$\mu m$]")
    axs[0].set_title(f"Phase shifted electric field on the mirror surface \n ({prop} propagation ) \n \
        Average phase = {phase_avg:.4f}, RMS phase = {phase_rms:.4f}\n \
        On-axis phase = {calc_phase(e_interp[grid_size//2]):.4f}, Paraxial phase = {-gouy:.4f}\n \
        alpha = {1/(k*z0):.4f}, L={L}, Rm={Rm}",fontsize=10)
    axs[0].set_ylabel(r"$E [V \mu m^{-1}]$")
    axs[0].plot(abs(x),(e_phase_shifted.real),'.',label=r'Re[$E$]')
    axs[0].plot(abs(x),(e_phase_shifted.imag),':',label=r'Im[$E$]')
    # axs[0].plot(abs(x),(100*e_phase_shifted.imag),color='orange',label=r'100*Im[$E$]')
    axs[0].plot(abs(x),abs(e_phase_shifted),'.',markersize=2,label=r'$|E|$')
    psi_expected=LG(p,l,X,Y,wL)[:,grid_size//2]
    axs[0].plot(abs(x),abs(psi_expected),'k--',label='Expected paraxial result',alpha=0.75)
    axs[0].legend(fontsize=8)

    axs[1].set_title(f"Phase of phase shifted electric field",fontsize=10)
    axs[1].set_ylabel(r'Arg $E$ [rad]')
    axs[1].plot(abs(x),calc_phase(e_phase_shifted),'r')
    plt.legend()
    plt.show()

else:
    # Propagation to plane
    e_L = propagate(Psi_0,x,L,mirror=mirror,paraxial=paraxial)# propagation to z=L, giving back electric field
    phase_L=calc_phase(e_L)
    phase_L_corrected = phase_L- k*(zm) # correction: subtracting the quadratic term.

    plt.figure()
    plt.title(f"Corrected phase of electric field at z=L plane \n ({prop} propagation)\n"+ \
              r"$\phi(E)-k\cdot z_m$")
    plt.xlabel(r'$x \;[\mu m]$')
    plt.ylabel(r"$Arg \;[rad]$")
    plt.axhline(-gouy,xmin=0,xmax=1,ls='--',color='r',label=f'Paraxial phase {-gouy:.4f}')
    plt.axhline(phase_L[grid_size//2],xmin=0,xmax=1,ls='--',color='k',label=rf'On-axis phase: {phase_L[grid_size//2]:.4f}')
    plt.plot(x[(x>=0)&(x<4.5)],phase_L_corrected[(x>=0)&(x<4.5)],label=r'$\psi(z=L)\cdot \exp(ikz_m)$')
    plt.legend()
    plt.show()

# #%%
# Ls = np.arange(2,11)
# Rs = np.arange(20,110,10)

# #%%
# results=np.zeros(7)
# for L in Ls:
#     for Rm in Rs:
#         # Setting paramters as in dimensions of the wavelength [micron]
#         wavelength = 1
#         k=2*PI/wavelength # Wave vector
#         z0 = np.sqrt(L*(Rm-L))
#         w0 = np.sqrt(z0*wavelength/PI)
#         p_wig = 1 # 0 for spherical mirror, 1 for parabloc mirror
#         n= 0;m=0;l=abs(n-m);p=min((n,m));N=p+l
#         # Grid discretization
#         grid_size = 2**8-1
#         xmax=8*wavelength
#         x=np.linspace(-xmax,xmax,grid_size);y=np.copy(x) # 1D arrays of Cartesian coordinates
#         z=np.linspace(0,L,grid_size) # z range from 0 to cavity length
#         X,Y=np.meshgrid(x,y) # 2D arrays of Cartesian coordinates
#         zm = Rm - np.sqrt(Rm**2-x**2) # 1D array containing the z 

#         Psi_0=LG(p,l,X,Y,w0)

#         zmin= L-Rm+np.sqrt(Rm**2-2*xmax**2) #determining a minimum z value from which to interpolate, based on the geometry of tha cavity
#         z_planes = np.linspace(zmin,L,4) #choose 4 z planes on the defined grid
#         e_interp = propagate(Psi_0,x,L-zm,paraxial=False,z_planes=z_planes) #propagation function, giving back the electric field
#         phase_avg,phase_rms=phase_moments(e_interp,abs(x))
#         # e_phase_shifted=e_interp*np.exp(-1j*phase_avg)
#         phase_on_axis=calc_phase(e_interp[grid_size//2])
#         alpha=1/(k*z0)
#         frac=L/(z0)
#         gouy=np.arctan2(L,z0)

#         nums = np.array([L,Rm,alpha,frac,-gouy,phase_on_axis,phase_avg])
#         results = np.vstack((results,nums))
# results=results[1:]
# results=np.around(results,4)
# #%%
# import pandas as pd 
# header=["L","Rm","alpha","L/(Rm-L)","parax","on-axis","avg"]
# df = pd.DataFrame(results,index=np.arange(0,81),columns=header)
# df.astype({"L": int, "Rm":int})
# df.to_csv("results.csv")
# df_sorted_a=df.sort_values(by='alpha',axis=0)
# #%%
# # #%%
# df_sorted_r=df.sort_values(by='L/(Rm-L)',axis=0)
# LRm=df_sorted_r['L/(Rm-L)']
# alpha=df_sorted_a['alpha']
# delta_phi=df_sorted_r['on-axis']-df_sorted_r['parax']

# plt.figure()
# plt.title('Difference between expected paraxial phase and on-axis phase \n \
#         as a function of '+ r'$\alpha$')
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$\Delta \phi$ [rad]')
# # plt.xscale('log')#;plt.yscale('log')
# plt.plot(alpha**2,delta_phi,'.')
# plt.show()


# plt.figure()
# plt.title('Difference between expected paraxial phase and on-axis phase \n \
#           as a function of '+ r'$\frac{L}{Rm}$')
# plt.xlabel(r'$\frac{L}{Rm}$')
# plt.ylabel(r'$\Delta \phi$ [rad]')
# plt.plot(LRm,delta_phi,'.')
# plt.show()
# # %%
# from scipy.optimize import curve_fit
# def quad(x,a,b,c):
#     return a*x**4+b*x**2+c

# def linear(x,a,b):
#     return a*x+b

# popt,pcov=curve_fit(linear,LRm,1000*delta_phi)

# plt.figure()
# plt.title('Difference between expected paraxial phase and on-axis phase \n \
#           as a function of '+ r'$\frac{L}{Rm}$' + '\n linear fit')
# plt.xlabel(r'$\frac{L}{Rm}$')
# plt.ylabel(r'$\Delta \phi$ [$10^{-3}$ rad]')
# plt.plot(LRm,delta_phi*1000,'.',label='Simulated points')
# plt.plot(LRm,linear(LRm,*popt),'r',label=rf'Fit: $\Delta \phi = {popt[0]:.3f}\cdot L/R_m + {popt[1]:.3f}$')
# plt.legend()
# plt.show()
# # %%

# popt,pcov=curve_fit(linear,alpha,1000*delta_phi)

# plt.figure()
# plt.title('Difference between expected paraxial phase and on-axis phase \n \
#           as a function of '+ r'$\alpha^2$' + '\n linear fit')
# plt.xlabel(r'$\log \left(\alpha^2\right)$')
# plt.ylabel(r'$\Delta \phi$ [$10^{-3}$ rad]')
# # plt.yscale('log')
# plt.plot(alpha,1000*delta_phi,'.',label='Simulated points')
# plt.plot(alpha,linear(alpha,*popt),'r',label=rf'Fit: $\Delta \phi = {popt[0]:.3f}\cdot \log \alpha^2 + {popt[1]:.3f}$')
# plt.legend()
# plt.show()
# %%
