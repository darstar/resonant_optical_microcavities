#%%
from objects import Geometry, Mode, Field,Field1D,Field2D
from plot_routines import plot_mirr, plot_plane
import numpy as np
import matplotlib.pyplot as plt 
PI=np.pi
#%%
wavelength=1
L=10
Rm=50
grid_size=2**8-1
pwig = 1
geometry = Geometry(wavelength,L,Rm,pwig,grid_size)
zm=geometry.mirror_coords()

l=0
p=2
paraxial=False

Psi_0=Mode(p,l,geometry,paraxial)

psi_0=Psi_0.field_profile().cross_section()
psi_expected=Psi_0.field_profile(L)
# %%
e_mirr = Psi_0.propagate_mirr()
e_plane=Psi_0.propagate_plane()
# %%
plot_mirr(e_mirr,Psi_0)
plot_plane(e_plane,Psi_0)

# %%
# # Modal decomposition of e_mirr
psi_mirr=e_mirr#.slowly_varying(zm)#Field1D(e_mirr * np.exp(-1j*geometry.k*geometry.mirror_coords()),geometry)

psi_1=Mode(1,l,geometry,paraxial).field_profile().cross_section()
psi_2=Mode(2,l,geometry,paraxial).field_profile().cross_section()
psi_3=Mode(3,l,geometry,paraxial).field_profile().cross_section()
weight = geometry.xmax/255*2*PI*abs(geometry.x)

alpha_0 = np.sum( (psi_mirr)*psi_0*weight)
alpha_1 = np.sum( (psi_mirr)*psi_1*weight)

alpha_2 = np.sum((psi_mirr)*psi_2*weight)
alpha_3 = np.sum((psi_mirr)*psi_3*weight)

# print(f"alpha0: {alpha_0}, abs: {abs(alpha_0)**2}")
print(f"alpha1: {alpha_1}, abs: {abs(alpha_1)**2}")
print(f"alpha2: {alpha_2}, abs: {abs(alpha_2)**2}")
print(f"alpha3: {alpha_3}, abs: {abs(alpha_3)**2}")

psi_decomposed = psi_1*alpha_1+psi_2*alpha_2+psi_3*alpha_3

plt.figure()
plt.title("Modal decomposition of the electric field at the mirror")
plt.xlabel(r"x [$\mu m$]");plt.ylabel(r"Abs(field) [$V\mu m^{-1}$]")
plt.plot(abs(geometry.x),psi_mirr.abs,label=r"$E_{mirr}$")
plt.plot(abs(geometry.x),psi_decomposed.abs,'r--',alpha=0.75,label=rf"{abs(alpha_0):.4f}LG$_0$$_0$+{abs(alpha_1):.4f}LG$_0$$_1$+{abs(alpha_2):.4f}LG$_0$$_2$")
plt.legend()
# %%
