#%%
from objects import Geometry, Mode, Field,Field1D,Field2D, projection,projection2D,modal_decomposition,expected_vals
from plot_routines import plot_mirr, plot_plane, plot_decomposition_mirr,plot_modes,plot_intensity
import numpy as np
import matplotlib.pyplot as plt 
import time
PI=np.pi
L=2.2 # cavity length
Rm=17.5# mirror radius of curvature
grid_size=2**11-1
#%%
results=np.load("simulations.npy",allow_pickle=True)
import pandas as pd
df = pd.DataFrame(data = results[1:], 
                  columns = results[0]).sort_values(['1/8k(Rm-L)','mirr/plane','l'])
# df=df.drop(223)
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
plt.errorbar(df[N0]["1/8k(Rm-L)"][(df[N0]["mirr/plane"]=="mirr")],diff0[(df[N0]["mirr/plane"]=="mirr")],yerr=df[N0]["rms phase"][(df[N0]["mirr/plane"]=="mirr")],marker='.',ls='none',label='Propagation correction')
plt.errorbar(df[N0]["1/8k(Rm-L)"][(df[N0]["mirr/plane"]=="plane")],diff0[(df[N0]["mirr/plane"]=="plane")],yerr=df[N0]["rms phase"][(df[N0]["mirr/plane"]=="plane")],marker='.',ls='none',label='Wavefront correction')
plt.plot(df[N0]["1/8k(Rm-L)"],(2)*df[N0]["1/8k(Rm-L)"],color='C0',label='Predicted')
plt.plot(df[N0]["1/8k(Rm-L)"],2*(2.2/df[N0]["Rm"])*df[N0]["1/8k(Rm-L)"],color='orange',label='Predicted')
plt.legend()
plt.savefig("Figures/N0_phase_difference_EB.pdf",bbox_inches="tight")
#%%
plt.figure()
plt.xlabel("Relative strength")
plt.ylabel(r"$\Delta \phi \; [rad]$")
# plt.xscale('log');plt.yscale('log')
plt.errorbar(df[N2]["1/8k(Rm-L)"][(df[N2]["mirr/plane"]=="mirr")][(df[N2]["p"]==1)],diff2[(df[N2]["mirr/plane"]=="mirr")][(df[N2]["p"]==1)],yerr=df[N2]["rms phase"][(df[N2]["mirr/plane"]=="mirr")][(df[N2]["p"]==1)],marker='.',ls='none',label='Propagation correction')
plt.errorbar(df[N2]["1/8k(Rm-L)"][(df[N2]["mirr/plane"]=="plane")][(df[N2]["p"]==1)],diff2[(df[N2]["mirr/plane"]=="plane")][(df[N2]["p"]==1)],yerr=df[N2]["rms phase"][(df[N2]["mirr/plane"]=="plane")][(df[N2]["p"]==1)],marker='.',ls='None',label='Wavefront correction')
plt.plot(df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==1)],(12)*df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==1)],color='orange',label='Predicted')
plt.plot(df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==1)],12*(2.2/df[N2]["Rm"][(df[N2]["p"]==1)])*df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==1)],color='C0',label='Predicted')
plt.legend()
# plt.savefig("Figures/N2_p1_phase_difference_EB.pdf",bbox_inches="tight")


#%%
plt.figure()
plt.xlabel("Relative strength")
plt.ylabel(r"$\Delta \phi \; [rad]$")
# plt.xscale('log');plt.yscale('log')
plt.errorbar(df[N1]["1/8k(Rm-L)"][(df[N1]["mirr/plane"]=="mirr")],diff1[(df[N1]["mirr/plane"]=="mirr")],yerr=df[N1]["rms phase"][(df[N1]["mirr/plane"]=="mirr")],marker='.',ls='none',label='Propagation correction')
plt.errorbar(df[N1]["1/8k(Rm-L)"][(df[N1]["mirr/plane"]=="plane")],diff1[(df[N1]["mirr/plane"]=="plane")],yerr=df[N1]["rms phase"][(df[N1]["mirr/plane"]=="plane")],marker='.',ls='none',label='Wavefront correction')
plt.plot(df[N1]["1/8k(Rm-L)"],(6)*df[N1]["1/8k(Rm-L)"],color='C0',label='Predicted')
plt.plot(df[N1]["1/8k(Rm-L)"],6*(2.2/df[N1]["Rm"])*df[N1]["1/8k(Rm-L)"],color='orange',label='Predicted')
plt.legend()
# plt.savefig("Figures/N1_phase_difference_EB.pdf",bbox_inches="tight")
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
from scipy.optimize import curve_fit
def quad(x,a):
    return a*x**2

popt0,pcov0 = curve_fit(quad,np.log10((df[N0]["1/8k(Rm-L)"].values.astype(float))),np.log10(diff0[(df[N0]["p"]==0)]))
popt1,pcov1 = curve_fit(quad,(np.log10(df[N1]["1/8k(Rm-L)"].values.astype(float))),np.log10(diff1[(df[N1]["p"]==0)]))
popt2,pcov2 = curve_fit(quad,np.log10((df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)].values.astype(float))),np.log10(diff2[(df[N2]["p"]==0)]))

plt.figure()
# plt.xlim(0.0038,-0.0003)
plt.xscale('log');plt.yscale('log')
# plt.xlim(10**(-2),10**(-5))
plt.xlabel("Relative strength")
plt.ylabel(r"$\Delta \phi \; [rad]$")
plt.plot((df[N0]["1/8k(Rm-L)"].values.astype(float)),diff0[(df[N0]["p"]==0)],'.',label=f'N=0')
plt.plot((df[N0]["1/8k(Rm-L)"].values.astype(float)),popt0*(df[N0]["1/8k(Rm-L)"].values.astype(float))**2,'--',color='C0')
plt.plot((df[N0]["1/8k(Rm-L)"].values.astype(float)),diff1[(df[N1]["p"]==0)].values-diff0[(df[N0]["p"]==0)].values,'.')
plt.plot((df[N0]["1/8k(Rm-L)"].values.astype(float)),diff2[(df[N2]["p"]==0)].values-diff0[(df[N0]["p"]==0)].values,'.')


# plt.plot((df[N1]["1/8k(Rm-L)"].values.astype(float)),diff1[(df[N1]["p"]==0)],'.',label=f'N=1')
# plt.plot((df[N1]["1/8k(Rm-L)"].values.astype(float)),popt1*(df[N1]["1/8k(Rm-L)"].values.astype(float))**2,'--',color='orange')

# plt.plot((df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)].values.astype(float)),diff2[(df[N2]["p"]==0)],'.',label=f'N=2')
# plt.plot((df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)].values.astype(float)),popt2*(df[N2]["1/8k(Rm-L)"][(df[N2]["p"]==0)].values.astype(float))**2,'--',color='green')

plt.legend()
# plt.savefig("paraxial_phase_difference.pdf",bbox_inches="tight")

#%%
# mask_pl=df[N0]["mirr/plane"]=="mirr"
mask4=(df[N2]["p"]==0)
mask1=(df[N2][mask4]["Rm"]==5.8)
mask2=(df[N2][mask4]["Rm"]==8.3)
mask3=(df[N2][mask4]["Rm"]==17.3)

mask = np.ma.mask_or(mask1,mask2)
mask=np.ma.mask_or(mask,mask3)
# %%
