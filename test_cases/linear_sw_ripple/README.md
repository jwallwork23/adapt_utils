### Simple linear shallow water problem

A reimplementation of the simple linear shallow water test case considered in my MRes project (found
[MRes][1]) using continuous space-time FEM.


### Parameters:

  * Equation set: non-rotational shallow water equations, linearised about state of rest
  * Spatial domain: [0, 4]x[0, 4]
  * Temporal domain: [0, 2.5]
  * Initial conditions: zero velocity and Gaussian elevation
  * Bathymetry:
    * Constant depth 0.1
    * Shelf break between depths 0.1 and 0.01
  * Boundary conditions: free-slip
  * Inviscid


[1]: https://github.com/jwallwork23/MResProject "here"
