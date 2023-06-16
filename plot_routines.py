from objects import Field,Field1D,Field2D,Geometry,Mode,modal_decomposition
import matplotlib.pyplot as plt
import numpy as np
PI=np.pi
def plot_intensity(field:Field2D,**kwargs):
    """
    Intensity plot of a 2D field profile
    """
    plt.xlabel(r"x [$\mu m$]");plt.ylabel(r"y [$\mu m$]")
    plt.imshow(field.intensity,cmap='jet',origin='lower', \
                   extent=[-field.geometry.xmax,field.geometry.xmax,-field.geometry.xmax,field.geometry.xmax],**kwargs)
    plt.colorbar()

def plot_mirr(e_mirr:Field1D,e_0:Mode):

    geometry=e_mirr.geometry
    mask = abs(geometry.x)<=geometry.xmax
    phase_avg,phase_rms=e_mirr.phase_moments()
    e_phase_shifted = e_mirr.phase_shift()
    e_expected = e_0.field_profile(geometry.L)

    if e_0.paraxial:
        paraxial='paraxial'
    else:
        paraxial='non-paraxial'
    if e_0.l!=0:
        e_mirr.phase_rectified[0]==0

    fig,axs = plt.subplots(2,1,sharex=True)
    fig.subplots_adjust(wspace=0.3)
    plt.xlabel(r"x [$\mu m$]")
    axs[0].set_title(f"Phase shifted  field on the mirror surface \n ({paraxial} propagation of {e_0.l}{e_0.p} mode) \n \
        Average phase = {phase_avg:.4f}, RMS phase = {phase_rms:.4f}\n \
        Paraxial phase = {e_0.get_gouy(geometry.L):.4f}\n \
        1/(8k(Rm-L))={1/(8*geometry.k*(geometry.Rm-geometry.L)):.5f}, L={geometry.L}, Rm={geometry.Rm}",fontsize=10)
    axs[0].set_title("(a)")
    axs[0].set_ylabel(r"$\psi [V \mu m^{-1}]$")
    axs[0].plot(abs(geometry.x[mask]),(e_phase_shifted.real[mask]),'.',label=r'Re[$\psi$]')
    axs[0].plot(abs(geometry.x[mask]),(e_phase_shifted.imag[mask]),':',label=r'Im[$\psi$]')
    axs[0].plot(abs(geometry.x[mask]),(10*e_phase_shifted.imag[mask]),color='orange',label=r'10*Im[$\psi$]')
    axs[0].plot(abs(geometry.x[mask]),e_phase_shifted.abs[mask],'.',color='green',markersize=2,label=r'$|\psi|$')
    axs[0].plot(abs(geometry.x[mask]),e_expected.cross_section().abs[mask],'k--',label='Exact paraxial result',alpha=0.75)
    axs[0].legend(fontsize=8)

    axs[1].set_title(fr"Phase profile",fontsize=10)
    axs[1].set_ylabel(r'Arg $\psi$ [rad]')
    axs[1].plot(abs(geometry.x[mask]),e_phase_shifted.phase_rectified[mask],'r')

    plt.legend(loc=1)
    plt.show()

def plot_plane(psi_e:Field1D,e_0:Mode):
    """
    Phase plot of the field propagation to z=L.
    """
    geometry=psi_e.geometry
    mask = (abs(geometry.x)<=geometry.xmax)
    e_phase = psi_e.phase_rectified
    phase_avg,phase_rms=psi_e.phase_moments()
    if e_0.paraxial:
        paraxial='paraxial'
    else:
        paraxial='non-paraxial'
    gouy = e_0.get_gouy(geometry.L)
    
    plt.title(r"Corrected phase of slowly-varying field at z=L plane"+"\n" +f"({paraxial} propagation of {e_0.l}{e_0.p})" +"\n"+ \
            f"Average phase = {phase_avg:.4f}, RMS phase = {phase_rms:.4f}"+"\n"+\
                f"Paraxial phase = {gouy:.4f}" "\n"+ \
        f"1/(8k(Rm-L)={1/(8*geometry.k*(geometry.Rm-geometry.L)):.5f}, L={geometry.L}, Rm={geometry.Rm}",fontsize=10)
    plt.xlabel(r'$x \;[\mu m]$')
    plt.ylabel(r"$Arg \psi \;[rad]$")
    plt.plot(abs(geometry.x[mask]),e_phase[mask])#,label=r'$\psi(z=L)\cdot \exp(ik\Delta z)$')
    plt.axhline(gouy,xmin=min(abs(geometry.x)),xmax=geometry.xmax,ls='--',color='r',label=f'Paraxial phase {gouy:.4f}')
    plt.axhline(phase_avg,xmin=min(abs(geometry.x)),xmax=geometry.xmax,ls='--',color='k',label=rf'Average phase: {phase_avg:.4f}')
    plt.legend()
    plt.savefig(f"Figures/paraxial{geometry.Rm}_fullx.pdf",bbox_inches="tight")
    plt.show()

def plot_decomposition(psi_e_mirr:Field1D,Psi_0:Mode):

    psi_e_mirr=psi_e_mirr.phase_shift()
    geometry=psi_e_mirr.geometry
    ps,alphas,e_decomposed = modal_decomposition(psi_e_mirr,Psi_0)

    l=Psi_0.l;p=Psi_0.p
    title_string="Modal decomposition of the phase shifted field at the curved mirror \n"
    for i in range(len(alphas)):
        if i == len(alphas)-1:
            title_string+=f"{abs(alphas[i])**2:.4f}"+rf"LG$_{l}$$_{ps[i]}$"
        else:
            title_string+=f"{abs(alphas[i])**2:.4f}"+rf"LG$_{l}$$_{ps[i]}$+"
    title_string+=f"\n {l}{p} mode at z=0"
    plt.figure()
    plt.title(title_string)
    plt.xlabel(r"x [$\mu m$]");plt.ylabel(r"Abs(field) [$V\mu m^{-1}$]")
    plt.plot(abs(geometry.x),psi_e_mirr.abs,'.',label=r"$\psi_{mirr}$")
    plt.plot(abs(geometry.x),e_decomposed[0].abs,'r',alpha=0.75,label="Decomposed field")
    plt.legend()
    plt.show()