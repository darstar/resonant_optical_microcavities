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
Field derived class, containing a 1D field profile, which is a 2D profile corss-sected at y=0. It has an extra feature: the on-axis-phase, which is the
ohase of the field at r=0,z=L.
#### Routines
power:
  calculates the 1D power, weighted over the intensity and using the polar jacobian. Fields are normalized, so this is a unitarty operation check
 
phase_moments:
  calculates the average and rms phase
  
 phase_shift:
  shifts the profile over the average phase.
 
 phase_plate:
  used for plane propagation, adds a phase plate factor which corrects the phase of the propagation,
  by adding a phase lag of k*zmirror.

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
rayleigh range z0 and beam width w0 are also calculated from these parameters.

### Routines:
field_profile:
   calculates the field profile, as a normalized LG mode at a z coordinate. The default is z=0, i.e the field
   at the flat mirror. If a different z-coordinate is given, for example z=L, it returns the paraaxial expectation 
   of the field at that z plane.
   
get_gouy:
  returns the gouy phase at a certain z coordinate.
  
angular_spectrum:
  propagates the initial field from z=0 to a different z plane.
  
propagate_mirr:

propagate_plane:


# Plot routines file
Creates plots depending on mirror/plane propagation.
