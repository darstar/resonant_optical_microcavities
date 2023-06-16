#%%
from objects import Geometry, Mode, Field,Field1D,Field2D, projection,modal_decomposition,expected_coupling
from plot_routines import plot_mirr, plot_plane
import numpy as np
import matplotlib.pyplot as plt 
import time
PI=np.pi
#%%
# Setting geometric paramters of the cavity
wavelength=0.633
L=2.2 # cavity length
Rm=17.5# mirror radius of curvature
grid_size=2**11-1

l=0
p=0
N=2*p+l
pwig=1 # mirror shape parameter
paraxial=True
geometry = Geometry(wavelength,L,Rm,pwig,grid_size,N)
Psi_0=Mode(p,l,N,geometry,paraxial) #Initial mode profile at the flat mirror

psi_e_mirr=Psi_0.propagate_mirr()
psi_e_phase_plate=Psi_0.propagate_plane()

ps,alphas,modes=modal_decomposition(psi_e_mirr,Psi_0)
print("Mirror propagation: ") # Display the modal decomposition amplitudes
for i in range(len(ps)):
    print("p=",ps[i]," abs(alpha)=",abs(alphas)[i]," Arg(alpha)=",np.angle(alphas*np.sign(alphas.real))[i])

# Plot the profiles for curved and plane mirror propagation
plot_mirr(psi_e_mirr,Psi_0)
plot_plane(psi_e_phase_plate,Psi_0)
# %%
