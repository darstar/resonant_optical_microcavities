#%%
from objects import Geometry, Mode, Field,Field1D,Field2D, projection,modal_decomposition#,composed_field_profile
from plot_routines import plot_mirr, plot_plane, plot_decomposition,plot_modes,plot_intensity
import numpy as np
import matplotlib.pyplot as plt 
PI=np.pi
#%%
# Setting geometric paramters of the cavity
wavelength=1 
L=10 # cavity length
Rm=50 # mirror radius of curvature
grid_size=2**8-1
pwig = 1 # mirror shape parameter

geometry = Geometry(wavelength,L,Rm,pwig,grid_size)
zm=geometry.mirror_coords() # mirror coordinates L-z(x,y)

l=0
p=0
paraxial=False

Psi_0=Mode(p,l,geometry,paraxial) # mode at the flat mirror
psi_0=Psi_0.field_profile().cross_section() # field profile of the mode at the flat mirror, at y=0
psi_expected=Psi_0.field_profile(L) # expected (paraxial) field profile at the curved mirror at z=L
# %%
e_mirr = Psi_0.propagate_mirr() # electric field propagated to the mirror
e_phase_plate=Psi_0.propagate_plane() # electric field propagated to the z=L plane
# %%
plot_mirr(e_mirr,Psi_0)
plot_plane(e_phase_plate,Psi_0)

