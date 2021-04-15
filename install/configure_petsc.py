#!/usr/bin/python
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath('config'))
    import configure
    configure_options = [
        'PETSC_ARCH=arch-adapt',
        '--with-shared-libraries=1',
        '--with-debugging=0',
        # '--with-fc=0',
        '--with-fortran-bindings=0',

        # '--with-metis=1',
        # '--with-metis-dir=/usr',
        # '--with-metis-include=/usr/include',
        # '--with-metis-lib=/usr/lib/x86_64-linux-gnu/libmetis.so',
        '--download-metis=1',

        # '--with-parmetis=1',
        # '--with-parmetis-dir=/home/joe/software/build/parmetis',
        '--download-parmetis=1',

        # '--download-hdf5',
        '--download-hdf5=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.bz2',
        # '--download-netcdf',   # NOTE: recently removed
        # '--download-pnetcdf',  # NOTE: recently removed
        '--download-scalapack',
        '--download-mumps',
        '--download-triangle',
        # '--download-ctetgen',
        '--download-chaco',
        '--download-hypre',
        # '--download-exodusii',
        '--with-zlib',

        # '--with-eigen=1',
        # '--with-eigen-dir=/usr/local',
        '--download-eigen',

        '--with-cxx-dialect=C++11',

        # '--download-libmesh',
        # '--with-pragmatic=1',
        # '--with-pragmatic-dir=/home/joe/forks/pragmatic',
        # '--with-pragmatic-include=/home/joe/software/pragmatic_with_vtk/build/include',
        # '--with-pragmatic-lib=/home/joe/software/pragmatic_with_vtk/build/lib/libpragmatic.so',
        '--download-pragmatic',
        # '--download-pragmatic-commit=75266ed005407030fa3317879d375a9328e436ae',
    ]
    configure.petsc_configure(configure_options)
