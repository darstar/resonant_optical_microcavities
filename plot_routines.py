from objects import Field,Field1D,Field2D,Geometry,Mode,modal_decomposition
import matplotlib.pyplot as plt
import numpy as np
PI=np.pi
def plot_intensity(field:Field2D,**kwargs):
    plt.xlabel(r"x [$\mu m$]");plt.ylabel(r"y [$\mu m$]")
    plt.imshow(field.intensity,cmap='jet',origin='lower', \
                   extent=[-field.geometry.xmax,field.geometry.xmax,-field.geometry.xmax,field.geometry.xmax],**kwargs)
    plt.colorbar()

def plot_mirr(e_mirr:Field1D,e_0:Mode):

    geometry=e_mirr.geometry
    phase_avg,phase_rms=e_mirr.phase_moments()
    e_phase_shifted = e_mirr.phase_shift()
    e_expected = e_0.field_profile(geometry.L)
    print(e_phase_shifted.phase[geometry.grid_size//2])
    if e_0.paraxial:
        paraxial='paraxial'
    else:
        paraxial='non-paraxial'

    fig,axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(wspace=0.3)
    plt.xlabel(r"x [$\mu m$]")
    axs[0].set_title(f"Phase shifted electric field on the mirror surface \n ({paraxial} propagation of {e_0.l}{e_0.p} mode) \n \
        Average phase = {phase_avg:.4f}, RMS phase = {phase_rms:.4f}\n \
        On-axis phase = {e_mirr.phase_on_axis:.4f}, Paraxial phase = {e_0.get_gouy(geometry.L):.4f}\n \
        L/(Rm-L)={geometry.L/(geometry.Rm-geometry.L)}, L={geometry.L}, Rm={geometry.Rm}",fontsize=10)
    
    axs[0].set_ylabel(r"$E [V \mu m^{-1}]$")
    axs[0].plot(abs(geometry.x),(e_phase_shifted.real),'.',label=r'Re[$E$]')
    axs[0].plot(abs(geometry.x),(e_phase_shifted.imag),':',label=r'Im[$E$]')

    # axs[0].plot(abs(geometry.x),(100*e_phase_shifted.imag),color='orange',label=r'100*Im[$E$]')
    axs[0].plot(abs(geometry.x),e_phase_shifted.abs,'.',markersize=2,label=r'$|E|$')
    
    axs[0].plot(abs(geometry.x),e_expected.cross_section().abs,'k--',label='Expected paraxial result',alpha=0.75)
    axs[0].legend(fontsize=8)

    axs[1].set_title(fr"Phase (mod $\pi$) of phase shifted electric field",fontsize=10)
    axs[1].set_ylabel(r'Arg $E$ [rad]')
    if e_0.p!=0:
        axs[1].plot(abs(geometry.x),np.mod(e_phase_shifted.phase,PI),'r')
    else:
        axs[1].plot(abs(geometry.x),e_phase_shifted.phase+PI,'r')

    plt.legend()
    plt.show()

def plot_plane(e_plane:Field1D,e_0:Mode):
    """
    Phase plot of the field propagation to z=L.
    """
    geometry=e_plane.geometry
    e_phase = e_plane.phase + geometry.k*(geometry.mirror_coords()-geometry.L)
    mask=(geometry.x>0)&(geometry.x<7)
    if e_0.paraxial:
        paraxial='paraxial'
    else:
        paraxial='non-paraxial'
    gouy = e_0.get_gouy(geometry.L)
    on_axis=e_plane.phase_on_axis

    plt.title(r"Corrected phase (mod $\pi$ ) of electric field at z=L plane"+"\n" +f"({paraxial} propagation of {e_0.l}{e_0.p}) \n"+ \
              r"$Arg[E]-k\cdot z_m$")
    plt.xlabel(r'$x \;[\mu m]$')
    plt.ylabel(r"$Arg \;[rad]$")
    plt.axhline(np.mod(gouy,PI),xmin=0,xmax=1,ls='--',color='r',label=f'Paraxial phase {np.mod(gouy,PI):.4f}')
    plt.axhline(np.mod(on_axis,PI),xmin=0,xmax=1,ls='--',color='k',label=rf'On-axis phase: {np.mod(on_axis,PI):.4f}')
    plt.plot(geometry.x[mask],np.mod(e_phase[mask],PI),label=r'$\psi(z=L)\cdot \exp(ikz_m)$')
    plt.legend()
    plt.show()

def plot_decomposition(e_mirr:Field1D,Psi_0:Mode):
    """
    Intensity plot of a 2D field profile
    """
    geometry=e_mirr.geometry
    ps,alphas,e_decomposed = modal_decomposition(e_mirr,Psi_0)

    l=Psi_0.l;p=Psi_0.p
    title_string="Modal decomposition of the electric field at the mirror \n"
    for i in range(len(alphas)):
        if i == len(alphas)-1:
            title_string+=f"{abs(alphas[i]):.4f}"+rf"LG$_{l}$$_{ps[i]}$"
        else:
            title_string+=f"{abs(alphas[i]):.4f}"+rf"LG$_{l}$$_{ps[i]}$+"
    title_string+=f"\n {l}{p} mode at flat mirror"
    plt.title(title_string)
    plt.xlabel(r"x [$\mu m$]");plt.ylabel(r"Abs(field) [$V\mu m^{-1}$]")
    plt.plot(abs(geometry.x),e_mirr.abs,'.',label=r"$E_{mirr}$")
    plt.plot(abs(geometry.x),np.sum(e_decomposed).abs,'r',alpha=0.75,label="Decomposed field")

    plt.legend()
 

def plot_modes(e_mirr,Psi_0):
    geometry=e_mirr.geometry
    ps,alphas,e_decomposed = modal_decomposition(e_mirr,Psi_0)
    plt.figure()
    plt.plot(abs(geometry.x),e_mirr.abs,'.',label=r"$E_{mirr}$")
    for i,mode in enumerate(e_decomposed):
        plt.plot(abs(geometry.x),mode.abs,alpha=0.5,ls='--',label = f"0{ps[i]}")
    plt.legend()
    plt.show()
