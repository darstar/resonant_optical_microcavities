#%%
from OOP import Geometry, Mode, Field, plot_mirr, plot_plane
import numpy as np
import matplotlib.pyplot as plt 
#%%
wavelength=1
L=10
Rm=50
grid_size=2**8-1
pwig = 1
geometry = Geometry(wavelength,L,Rm,pwig,grid_size)

l=0
p=1
paraxial=False

Psi_0=Mode(p,l,geometry,paraxial)
psi_0=Psi_0.field_profile().cross_section()
psi_expected=Psi_0.field_profile(L)
# %%
psi_mirr = Psi_0.propagate_mirr()
psi_plane=Psi_0.propagate_plane()
# %%
plot_mirr(psi_mirr,Psi_0,geometry)
plot_plane(psi_plane,Psi_0,geometry)
# %%