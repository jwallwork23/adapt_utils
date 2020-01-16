### Mesh adaptation in Firedrake

In this code, anisotropic and isotropic goal-oriented mesh adaptation is applied to solving shallow
water and tracer transport problems using the coastal, estuarine and ocean modelling solver provided
by [Thetis][2]. Thetis is built upon the [Firedrake][1] project, which enables the efficient
solution of FEM problems in Python by automatic generation of [PETSc][3] code. Anisotropic mesh
adaptation is achieved using [PRAgMaTIc][4]. A continuous adjoint solver is provided for advection-diffusion problems and the discrete adjoint code [Pyadjoint][5] can be used to generate adjoint
solutions for more general problems. This is research of the Applied Modelling and Computation Group
([AMCG][6]) at Imperial College London.

### Versions

* `v1.0`: 'Anisotropic Goal-Oriented Mesh Adaptation in Firedrake': [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3358565.svg)](https://doi.org/10.5281/zenodo.3358565)

* `v1.1`: 'Goal-Oriented Error Estimation and Mesh Adaptation for Shallow Water Modelling': [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3571867.svg)](https://doi.org/10.5281/zenodo.3571867)

### User instructions

* Clone this repository and make it accessible to the `PYTHONPATH` environment variable.
* If using `v1.0`, download Firedrake, PETSc and Pragmatic using [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3250888.svg)](https://doi.org/10.5281/zenodo.3250888).
* If using `v1.1`:
	* Install the same PETSc version as used in `v1.0`.
	* Download Firedrake using [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3568997.svg)](https://doi.org/10.5281/zenodo.3568997), with the flags `--install thetis --install pyadjoint --honour-petsc-dir`.

#### For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[My personal webpage][7]

[1]: http://firedrakeproject.org/ "Firedrake"
[2]: http://thetisproject.org/index.html "Thetis"
[3]: https://www.mcs.anl.gov/petsc/ "PETSc"
[4]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[5]: https://bitbucket.org/dolfin-adjoint/pyadjoint/src "Pyadjoint"
[6]: http://www.imperial.ac.uk/earth-science/research/research-groups/amcg/ "AMCG"
[7]: http://www.imperial.ac.uk/people/j.wallwork16 "My personal webpage."
