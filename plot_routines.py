from objects import Field,Field1D,Field2D,Geometry,Mode
import matplotlib.pyplot as plt
import numpy as np

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
    axs[0].set_title(f"Phase shifted electric field on the mirror surface \n ({paraxial} propagation of {e_0.l}{e_0.p} mode ) \n \
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

    axs[1].set_title(f"Phase of phase shifted electric field",fontsize=10)
    axs[1].set_ylabel(r'Arg $E$ [rad]')
    axs[1].plot(abs(geometry.x),e_phase_shifted.phase,'r')
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
    plt.figure()
    plt.title(f"Corrected phase of electric field at z=L plane \n ({paraxial} propagation)\n"+ \
              r"$Arg[E]-k\cdot z_m$")
    plt.xlabel(r'$x \;[\mu m]$')
    plt.ylabel(r"$Arg \;[rad]$")
    plt.axhline(e_0.get_gouy(geometry.L),xmin=0,xmax=1,ls='--',color='r',label=f'Paraxial phase {-e_0.get_gouy(geometry.L):.4f}')
    plt.axhline(e_plane.phase_on_axis,xmin=0,xmax=1,ls='--',color='k',label=rf'On-axis phase: {e_plane.phase_on_axis:.4f}')
    plt.plot(geometry.x[mask],e_phase[mask],label=r'$\psi(z=L)\cdot \exp(ikz_m)$')
    plt.legend()
    plt.show()

def plot_intensity(field:Field2D,**kwagrs):
    """
    Intensity plot of a 2D field profile
    """
    plt.xlabel(r"x [$\mu m$]");plt.ylabel(r"y [$\mu m$]")
    plt.imshow(field.intensity,cmap='jet',origin='lower', \
                   extent=[-field.geometry.xmax,field.geometry.xmax,-field.geometry.xmax,field.geometry.xmax],**kwargs)
    plt.colorbar()