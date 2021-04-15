## 15 turbine array

Test case as described in [Divett et al.].

Many thanks to [Nicolas Barral][1] for assisting with implementing this test case in Firedrake.

### Original problem
  * Unsteady shallow water in the presence of a 5 x 3 array of tidal turbines.
  * Rectangular domain [-1500, 1500] x [-500, 500].
  * Turbines are also parametrised as rectangular, with length 20 m and width 5 m.
  * Simple tidal forcings on the left hand and right hand boundaries which are exactly out of phase.
  * On the top and bottom boundaries, free-slip conditions are assumed.
  * Constant water depth of 50 m.
  * A reduced tidal period of 1.24 h is used (10% of the M2 tidal period).
  * Background (quadratic) friction coefficient 0.0025.
  * CFL-dependent timestep, with maximum permitted value 5 s.

### Differences in this implementation
  * Constant timestep of 3 s.
  * Horizontal (kinematic) viscosity 3 m^2 s^{-1}.

### Quantity of interest
Power output (integral over turbine footprints and time period of cubed fluid speed, scaled by
turbine drag coefficient and divided by water depth).

### References

[Divett et al.] T. Divett, R. Vennell, and C. Stevens. "Optimization of multiple turbine arrays in a
channel with tidally reversing flow by numerical modelling with adaptive mesh." Philosophical
Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 371.1985
(2013): 20120251.

[1]: https://nicolasbarral.fr "Nicolas Barral"
