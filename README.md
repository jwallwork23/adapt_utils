## Mesh adaptation for coastal ocean modelling in Firedrake

In this code, anisotropic and isotropic goal-oriented mesh adaptation is applied to solving a variety
of 2D coastal ocean modelling problems using the coastal, estuarine and ocean modelling solver
provided by [Thetis][2]. Thetis is built upon the [Firedrake][1] project, which enables the efficient
solution of FEM problems in Python by automatic generation of [PETSc][3] code.

Currently supported model components:
  * Shallow water equations (i.e. depth-averaged hydrodynamics);
  * Passive tracer transport (both non-conservative and conservative options available);
  * Sediment transport;
  * Exner equation.

Anisotropic mesh adaptation based on Riemannian metric fields is achieved using [PRAgMaTIc][4]. This
code provides a wide range of utilities for metric-based adaptation, including Hessian recovery,
metric combination and (isotropic and anisotropic) goal-oriented error estimators and metrics.
Mesh movement is also supported using hand-coded Firedrake-based Monge-Ampere and ALE solvers.

Continuous adjoint solvers are provided for shallow water and advection-diffusion problems. The
discrete adjoint code [Pyadjoint][5] can also be used to generate adjoint solutions for more general
problems.

This is research of the Applied Modelling and Computation Group ([AMCG][6]) at Imperial College
London.


### Publications and associated versions and test cases

  * J. G. Wallwork, N. Barral, D. A. Ham, M. D. Piggott, <em>'Anisotropic Goal-Oriented Mesh Adaptation in Firedrake'</em>, In: 28th International Meshing Roundtable, pp.83-100, (2020). DOI: 10.5281/zenodo.3653101.
    * Paper: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3653101.svg)](https://doi.org/10.5281/zenodo.3653101), URL: https://doi.org/10.5281/zenodo.3653101.
    * Code: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3358565.svg)](https://doi.org/10.5281/zenodo.3358565) (`v1.0`).
    * Test cases: `steady/test_cases/point_discharge2d`, `steady/test_cases/point_discharge3d`

  * J. G. Wallwork, N. Barral, S. C. Kramer, D. A. Ham, M. D. Piggott, <em>'Goal-Oriented Error Estimation and Mesh Adaptation for Shallow Water Modelling'</em>, Springer Nature Applied Sciences, volume 2, pp.1053--1063 (2020).
    * Paper: DOI:10.1007/s42452-020-2745-9, URL: https://rdcu.be/b35wZ.
    * Code: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3695801.svg)](https://doi.org/10.5281/zenodo.3695801) (`v1.3`).
    * Test case: `steady/test_cases/turbine_array`


### User instructions

  * Clone this repository and make it accessible to the `$PYTHONPATH` environment variable.
  * Set the `$SOFTWARE` environment variable to where you would like your PETSc and Firedrake installations to exist.
  * Copy the contents of the `install` directory into `$SOFTWARE` and enter that directory.
  * Call `bash install_petsc.sh` and then `bash install_firedrake.sh`, modifying these scripts, if desired. If installing on a fresh Ubuntu OS then you will need to call `bash install_compilers.sh` beforehand.
  * Once you have a working Firedrake installation, get to grips with `adapt_utils` by looking at the test cases in `steady/test_cases` and `unsteady/test_cases`, as well as the notebooks hosted [here][7].


#### For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[1]: http://firedrakeproject.org/ "Firedrake"
[2]: http://thetisproject.org/index.html "Thetis"
[3]: https://www.mcs.anl.gov/petsc/ "PETSc"
[4]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[5]: https://bitbucket.org/dolfin-adjoint/pyadjoint/src "Pyadjoint"
[6]: http://www.imperial.ac.uk/earth-science/research/research-groups/amcg/ "AMCG"
[7]: https://github.com/jwallwork23/adapt_utils_notebooks "adapt_utils_notebooks"
