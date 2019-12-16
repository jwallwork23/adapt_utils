## Two turbines in a channel


Test case as described in [1], modelling two turbines positioned in a channel. Two different
configurations are considered: one in which the turbines are aligned and another where they are
offset.


### Parameters:
  * Domain: [0, 1000]x[0, 300]
  * Viscosity coefficient: 1.0
  * Water depth: 40.0
  * Drag coefficient: 0.0025
  * Turbine locations:
    * Aligned case: (356, 150), (644, 150)
    * Offset case: (356, 132), (644, 118)
  * Turbine thrust coefficient corrected as recommended in [2].

### Boundary conditions:
  * Dirichlet u=(3, 0) on inflow x=0
  * Dirichlet eta=0 on outflow x=1000
  * Free-slip u.n=0 elsewhere

### Quantity of interest:
Power output (integral of cubed fluid speed integrated over turbine footprints, scaled by turbine
drag coefficient).


[1] Wallwork, J.G., Barral, N., Stephan, S.C., Ham, D.A., Piggott, M.D.: Goal-oriented error
    estimation and mesh adaptation for shallow water modelling (2019). (in preparation)

[2] Kramer, S.C. and Piggott, M.D., "A correction to the enhanced bottom drag parameterisation of
    tidal turbines." Renewable Energy 92 (2016): 385-396.
