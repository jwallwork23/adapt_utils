## 15 turbine array

Test case as described in [Divett et al.].

Many thanks to [Nicolas Barral][1] for assisting with implementing this test case in Firedrake.

### Parameters:
  * TODO

### Boundary conditions:
  * Simple tidal forcings on the left hand and right hand boundaries which are exactly out of phase.
  * On the top and bottom boundaries, free-slip conditions are assumed.

### Quantity of interest
Power output (integral over turbine footprints and time period of cubed fluid speed, scaled by
turbine drag coefficient and divided by water depth).

### References

[Divett et al.] T. Divett, R. Vennell, and C. Stevens. "Optimization of multiple turbine arrays in a
channel with tidally reversing flow by numerical modelling with adaptive mesh." Philosophical
Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 371.1985
(2013): 20120251.

[1]: https://nicolasbarral.fr "Nicolas Barral"
