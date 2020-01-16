### "Point Discharge with Diffusion"

Test case as described in TELEMAC-2D validation document version 7.0 [Riadh et al.].


### Parameters:
  * Domain: [0, 50]x[0, 10]
  * Fluid velocity: (1.0, 0.0)
  * Diffusivity coefficient: 0.1
  * Source location: (2.0, 5.0)
  * Delta function parametrisation:
    * Centred case: 0.07980
    * Offset case: 0.07972

### Boundary conditions:
  * Dirichlet zero on inflow x=0
  * Outflow condition on x=50
  * Neumann zero elsewhere

### Quantity of interest:
Integral of tracer concentration over region of interest, given by a circle of radius 0.5, centred at
  * Centred case: (20.0, 5.0)
  * Offset case: (20.0, 7.5)


[Riadh et al.] A. Riadh, G. Cedric, M. Jean, "TELEMAC modeling system: 2D hydrodynamics TELEMAC-2D
software release 7.0 user manual." Paris:  R&D, Electricite de France, p. 134 (2014).
