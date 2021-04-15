## Spaceship tidal turbine test case:

Original problem, as described in [Walkington & Burrows], models a tidal stream farm in a narrow
region between basins. The idealised domain is inspired by the Strangford Lough geometry, with a
tidal free surface forcing on the open boundary.

The later work of [Kramer et al.] compared Fluidity and MIKE21 using the same domain but a modified
bathymetry. It is this latter bathymetry which is used here.

Many thanks to [Nicolas Barral][1] and [Stephan Kramer][2] for assisting with implementing this test
case in Firedrake.

### Parameters:
  * TODO

### Boundary conditions:
  * The semicircular boundary is forced using the (spatially uniform) tidal forcing given in `resources/forcing/forcing.dat`.
  * In addition, we use a viscosity sponge to this boundary, increasing from 5 to 100. There are two options for the sponge type: linear and exponential.
  * On all other boundaries, free-slip conditions are assumed.

### Quantity of interest
Power output (integral over turbine footprints and time period of cubed fluid speed, scaled by
turbine drag coefficient and divided by water depth).

### References

[Walkington & Burrows] I. Walkington, R. Burrows, "Modelling tidal stream power potential", Applied
ocean research, 31:239–245, (2009).

[Kramer et al.] S. C. Kramer, M. D. Piggott, J. Hill, L. Kregting, D. Pritchard, B. Elsaesser,
"The modelling of tidal turbine farms using multi-scale unstructured mesh models", Proceedings of
the 2nd International Conference on Environmental Interactions of Marine Renewable Energy
Technologies (2014).

[1]: https://nicolasbarral.fr "Nicolas Barral"
[2]: https://www.imperial.ac.uk/people/s.kramer "Stephan Kramer"
