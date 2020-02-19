## Two turbines in a channel

Test case as described in [Wallwork et al.], modelling two turbines positioned in a channel. Two
different configurations are considered: one in which the turbines are aligned and another where
they are offset.

Many thanks to [Stephan Kramer][1] for coding the original fixed mesh version of this test case in
Firedrake.


### Parameters:
  * Domain: [0, 1000]x[0, 300]
  * Viscosity coefficient: 1.0
  * Water depth: 40.0
  * Drag coefficient: 0.0025
  * Turbine locations:
    * Aligned case: (356, 150), (644, 150)
    * Offset case: (356, 132), (644, 118)
  * Turbine thrust coefficient corrected as recommended in [Kramer and Piggott].

### Boundary conditions:
  * Dirichlet u=(3, 0) on inflow x=0
  * Dirichlet eta=0 on outflow x=1000
  * Free-slip u.n=0 elsewhere

### Quantity of interest:
Power output (integral over turbine footprints of cubed fluid speed, scaled by turbine
drag coefficient and divided by water depth).


### References

[Wallwork et al.] J.G. Wallwork, N. Barral, S.C. Kramer, D.A. Ham, M.D. Piggott, "Goal-oriented error
estimation and mesh adaptation for shallow water modelling" (2019). (submitted to Springer Applied
Natural Sciences)

[Kramer and Piggott] S.C. Kramer, M.D. Piggott, "A correction to the enhanced bottom drag
parameterisation of tidal turbines." Renewable Energy 92 (2016): 385-396.

[1]: https://www.imperial.ac.uk/people/s.kramer "Stephan Kramer"
