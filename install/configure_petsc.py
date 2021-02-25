#!/usr/bin/python
"""
Python script for setting PETSc configure options and then running the configuration.

Modified from the automatically generated script found in `$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/`.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('config'))
import configure


configure_options = [
    'PETSC_ARCH=arch-adapt',
    '--with-shared-libraries=1',
    '--with-debugging=0',
    '--with-fortran-bindings=0',
    '--with-zlib',
    '--with-cxx-dialect=C++11',

    '--download-metis=1',
    '--download-parmetis=1',

    # '--download-hdf5',
    '--download-hdf5=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.bz2',

    '--download-scalapack',
    '--download-mumps',
    '--download-triangle',
    # '--download-ctetgen',
    '--download-chaco',
    '--download-hypre',
    # '--download-exodusii',
    '--download-eigen',
    '--download-pragmatic',
]
configure.petsc_configure(configure_options)
