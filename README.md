### Mesh adaptation in Firedrake

In this code, anisotropic mesh adaptivity is applied to solving the nonlinear shallow water equations and advection-diffusion equations in the coastal, estuarine and ocean modelling solver provided by [Thetis][2]. The Thetis project is built upon the [Firedrake][1] project, which enables efficient FEM solution in Python by automatic generation of C code. Anisotropic mesh adaptivity is achieved using [PRAgMaTIc][3]. This is research of the Applied Modelling and Computation Group ([AMCG][4]) at
Imperial College London.

### Versions

* `v1.0`: 'Anisotropic Goal-Oriented Mesh Adaptation in Firedrake': [![DOI](https://zenodo.org/badge/169627287.svg)](https://zenodo.org/badge/latestdoi/169627287)

### User instructions

* If using `v1.0`, download Firedrake, PETSc and Pragmatic using [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3250888.svg)](https://doi.org/10.5281/zenodo.3250888).

For the current development version, follow the following instructions:

* Clone this repository and make it accessible to the PYTHONPATH environment variable.

* Download the [Firedrake][1] install script, set
    * ``export PETSC_CONFIGURE_OPTIONS="--download-pragmatic --with-cxx-dialect=C++11"``

    and install with option parameter ``--install thetis``.

* Fetch and checkout the remote branches
    * ``https://github.com/thetisproject/thetis/tree/joe`` for thetis;
    * ``https://github.com/taupalosaurus/firedrake`` for firedrake, fork ``barral/meshadapt``
    and call ``make`` in ``firedrake/src/firedrake`` to enable pragmatic drivers.


#### For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[1]: http://firedrakeproject.org/ "Firedrake"
[2]: http://thetisproject.org/index.html "Thetis"
[3]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[4]: http://www.imperial.ac.uk/earth-science/research/research-groups/amcg/ "AMCG"

