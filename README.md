# resonant_optical_microcavities

BSc thesis simulations on short optical microcavities. Part of Leiden University, Quantum Optics group. This code simulates 
how a Laguerre-Gauss mode propagates through a cavity, using the angular spectrum method. 

# Objects file
## Geometry
 ! All length units are in microns !
 
This object determines the geometry of the problem. It consists of the wavelength, of the cavity, radius of curvature, a factor pwig that determines the
shape of the mirror and a grid size.
From this, a grid for the x coordinates is made. 
Extra feature possibilities: asymmetric mirrors, with a different radius of curvature in the x and y direction.

### Routines
get_2D_grid:
 Returns a 2D cartesian grid, which is symmetric in the x and y direction.

get_mirror_coords:
 returns the mirror's z coordinates

get_mirror_planes:
  returns 4 evenly spaced z-planes around the mirror surface, used in th epropagation routines (explained later)
 
## Field
Base class, containing a geometry and field profile. The absolute value, intensity, real and imaginary part and phase are defined

### Field 1D
Field derived class, containing a 1D field profile, which is a 2D profile corss-sected at y=0.
#### Routines
power:
  calculates the 1D power, weighted over the intensity. Fields are normalized, so this is a unitarty operation check
 
phase_moments:
  calculates the intensity-weighted average and rms phase
  
 phase_shift:
  shifts the profile over the average phase.
 
 phase_plate:
  used for plane propagation, adds a phase plate factor which corrects the phase of the propagation,
  by adding a phase lag of k*zm.

### Field2D
Field derived class , containing a 2D field profile.
#### Routines:
power:
  calculates the power. Fields are normalized, so this is a unitarty operation check
cross section:
  returns a 1D field, which is the 2D field cross-sected at y=0.

## Mode
Paraxial mode in the LG (orthonormal) basis.
Defined by p and l modal numbers, a geometry and paraxial/non-paraxial propagation. The modal number N,
Rayleigh range z0 and beam width w0 are also calculated from these parameters.

### Routines:
field\_profile:
   calculates the field profile, as a normalized LG mode at a z coordinate. The default is z=0, i.e the field
   at the flat mirror. If a different z-coordinate is given, for example z=L, it returns the paraxial expectation 
   of the field at that z plane.
   
get\_gouy:
  returns the gouy phase at a certain z coordinate.

FFT2D:
  propagates the initial field from z=0 to a different z plane, using the angular spectrum method.
  
propagate\_mirr:
  propagates the initial field, using the angular spectrum method, to the mirror surface, by interpolation.

propagate\_plane:
 propagates the initial field, using the angular spectrum method, to the z=L plane and used a phase plate to correct for the mirror.

## Other routines outside classes
### projection
Projects the field profile on the mirror surface onto a
different p mode.
### modal decomposition
 Calculates the projection of the electric field on the mirror
to the LG{lp} basis. (up to delta p =+/-2)
Returns the values of p, alpha and the projected modes.
### expected\_coupling
Returns an array of the expected mode coupling for a certain (p,l) mode and relative strength 1/8k(Rm-L)

# Plot routines file
Creates plots depending on mirror/plane propagation.
###plot\_intensity
Plots the intensity of a 2D field object with colorbar.

## plot\_mirr
Creates plots of the field profile for propagation to the curved mirror and the corrected phase as in Figures 4.4, 4.5 and 4.6.

## plot\_plane
Creates a plot of the corrected phase for propagation to a plane mirror, as in Figure 4.2.

## plot\_decomposition
Plots the modal decomposition of a field profile as well as the field profile itself for comparison.

# Main file
This file contains the main program, from which the simulations are run. It defines the geometry parameters wavelength, L, Rm, grid\_size and pwig. You can also select the desired (p,l) mode and a Mode object is created from this. A paraxial variable is created to turn the paraxial propagation ON/OFF. The program computes the curved and plane mirror propagation and calculates the modal decomposition.
