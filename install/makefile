all: firedrake

compilers:
	source install_compilers.sh

petsc: compilers
	source install_petsc.sh

firedrake: petsc
	source install_firedrake.sh

basic: compilers
	source install_firedrake_no_adapt.sh
