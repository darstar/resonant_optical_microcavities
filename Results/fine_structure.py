#%%
from objects import Geometry, Mode, Field,Field1D,Field2D, projection,projection2D,modal_decomposition,expected_vals
from plot_routines import plot_mirr, plot_plane, plot_decomposition_mirr,plot_modes,plot_intensity
import numpy as np
import matplotlib.pyplot as plt 
import time
PI=np.pi
wavelength=0.633
L=2.2 # cavity length
Rm=17.5# mirror radius of curvature
grid_size=2**11-1
#%%
results=np.load("simulations.npy",allow_pickle=True)
import pandas as pd
df = pd.DataFrame(data = results[1:], 
                  columns = results[0]).sort_values(['1/8k(Rm-L)','mirr/plane','l'])
#%%
parax = (df['paraxial']==True)&(df['p']<2)&(df['l']<3)
N0 = (df['paraxial']==False)&(df['p']==0)&(df['l']==0)
N1 = (df['paraxial']==False)&(df['p']==0)&(df['l']==1)
N2 = (df['paraxial']==False)&(2*df['p']+df['l']==2)
N=0
phase0 = df['avg phase'][N0]
gouy0=-np.arctan(np.sqrt(2.2)/(df[N0]['Rm']-2.2).pow(1/2).values.astype(float))
diff0=-(np.mod(df[N0]['avg phase']-gouy0,PI)-PI)

gouy1=-np.arctan(np.sqrt(2.2)/(df[N1]['Rm']-2.2).pow(1/2).values.astype(float))*2
diff1=np.mod(gouy1-df[N1]['avg phase'],PI/2)

gouy2=-np.arctan(np.sqrt(2.2)/(df[N2]['Rm']-2.2).pow(1/2).values.astype(float))*3
diff2=np.mod(gouy2-df[N2]['avg phase'],PI)

plt.figure()
plt.xlabel("Relative strength")
plt.ylabel(r"$\Delta \phi \; [rad]$")
# plt.xscale('log');plt.yscale('log')
plt.plot(df[N2]["1/8k(Rm-L)"][(df[N2]["mirr/plane"]=="mirr")][(df[N2]["p"]==0)],diff2[(df[N2]["mirr/plane"]=="mirr")][(df[N2]["p"]==0)],'.',label='Propagation correction')
plt.plot(df[N2]["1/8k(Rm-L)"][(df[N2]["mirr/plane"]=="mirr")][(df[N2]["p"]==0)],diff2[(df[N2]["mirr/plane"]=="plane")][(df[N2]["p"]==0)],'.',label='Wavefront correction')
plt.plot(df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)],(12)*df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)],color='C0',label='Predicted')
plt.plot(df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)],1*(1+2.2/df[N2]["Rm"][(df[N2]["p"]==0)])*df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)],color='orange',label='Predicted')
plt.legend()
# plt.savefig("Figures/N2_p0_phase_difference.pdf",bbox_inches="tight")

plt.figure()
plt.xlabel("Relative strength")
plt.ylabel(r"$\Delta \phi \; [rad]$")
# plt.xscale('log');plt.yscale('log')
plt.plot(df[N1]["1/8k(Rm-L)"][(df[N1]["mirr/plane"]=="mirr")],diff1[(df[N1]["mirr/plane"]=="mirr")],'.',label='Propagation correction')
plt.plot(df[N1]["1/8k(Rm-L)"][(df[N1]["mirr/plane"]=="mirr")],diff1[(df[N1]["mirr/plane"]=="plane")],'.',label='Wavefront correction')
plt.plot(df[N1]["1/8k(Rm-L)"],(6)*df[N1]["1/8k(Rm-L)"],color='C0',label='Predicted')
plt.plot(df[N1]["1/8k(Rm-L)"],6*(1+2.2/df[N1]["Rm"])*df[N1]["1/8k(Rm-L)"],color='orange',label='Predicted')
plt.legend()
# plt.savefig("Figures/N1_phase_difference.pdf",bbox_inches="tight")
#%%
N0 = (df['paraxial']==True)&(df['p']==0)&(df['l']==0)
N1 = (df['paraxial']==True)&(df['p']==0)&(df['l']==1)
N2 = (df['paraxial']==True)&(2*df['p']+df['l']==2)
phase0 = df['avg phase'][N0]
gouy0=-np.arctan(np.sqrt(2.2)/(df[N0]['Rm']-2.2).pow(1/2).values.astype(float))
diff0=-(np.mod(df[N0]['avg phase']-gouy0,PI)-PI)

gouy1=-np.arctan(np.sqrt(2.2)/(df[N1]['Rm']-2.2).pow(1/2).values.astype(float))*2
diff1=np.mod(gouy1-df[N1]['avg phase'],PI/2)

gouy2=-np.arctan(np.sqrt(2.2)/(df[N2]['Rm']-2.2).pow(1/2).values.astype(float))*3
diff2=np.mod(gouy2-df[N2]['avg phase'],PI)

#%%
plt.figure()
plt.yscale('log');plt.xscale('log')
plt.xlabel("Relative strength")
plt.ylabel(r"$\Delta \phi \; [rad]$")
plt.plot(df[N0]["1/8k(Rm-L)"],diff0[(df[N0]["p"]==0)],'.',label='N=0')
plt.plot(df[N1]["1/8k(Rm-L)"],diff1[(df[N1]["p"]==0)],'.',label='N=1')
plt.plot(df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)],diff2[(df[N2]["p"]==0)],'.',label='N=2')
plt.legend()
#%%
def quad(x,a,b,c):
    return a*x**2+b*x+c
from scipy.optimize import curve_fit

x=df[N0]["1/8k(Rm-L)"].values.astype(float)
y=diff0[(df[N0]["p"]==0)].values.astype(float)

popt,pcov=curve_fit(quad,x,y)

# %%
