#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# Implementation of the scalar, paraxial solution to the Helmholz-like equation
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

def fundamental(X,Y,w0):
    return 1j/w0*np.exp(-(X**2+Y**2)/w0**2)

from scipy.special import assoc_laguerre, eval_hermitenorm, factorial,eval_genlaguerre
def LG(n,m,X,Y,w0,gouy):
    N=n+m
    l=np.abs(n-m)
    p=min((n,m))
    normalization = np.sqrt(2/np.pi) *factorial(p) / np.sqrt((factorial(n)*factorial(m)))
    scaling = np.sqrt(2)/w0
    r=X**2+Y**2
    rho = scaling**2*r
    theta=np.arctan2(Y,X)

    # f = (r_norm)**(l/2)*assoc_laguerre(r_norm,p,l)
    # Question: what to do with Gouy phase?? 
    normalization = (-1)**p* np.sqrt(factorial(p)/(np.pi*factorial(p+l)) )# mode normalization
    field = (rho)**(l)*eval_genlaguerre(p,l,rho**2) * np.exp(-rho**2/2) * normalization*np.exp(1j*l*theta)
    return field


def HG(n,m,X,Y,w):
    N=n+m
    scaling = np.sqrt(2)/w
    normalization = np.sqrt(2/np.pi) * 1/(np.sqrt(factorial(n)*factorial(m))) * 2**(-N/2)
    return normalization*eval_hermitenorm(n,scaling*X)*eval_hermitenorm(m,scaling*Y)*fundamental(X,Y,np.sqrt(2)*w0)

# radial_profile=LG(p,l,X,Y,w0)#,gouy)
# fund = fundamental(X,Y,w0)
fig,ax=plt.subplots(3,3)
fig.subplots_adjust(wspace=0,hspace=100)
for p in range(3):
    for l in range(3):
        radial_profile=LG(p,l,X,Y,w0,gouy)
        ax[p][l].axis('off')
        ax[p][l].set_title(f"{l}{p}")
        ax[p][l].imshow(np.abs(radial_profile)**2,cmap='binary')

# plt.colorbar()
plt.tight_layout()# %%
plt.savefig("LG.pdf",bbox_inches="tight")
# %%
