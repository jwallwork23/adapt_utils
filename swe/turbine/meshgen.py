import os


__all__ = ["generate_geo_file"]


def generate_geo_file(op, level='xcoarse', filepath='.'):
    """
    Generate domain geometry file using the specifications in `op`.

    :arg op: Parameter class.
    :kwarg level: Desired level of mesh resolution.
    :kwarg filepath: Where to save .geo file.
    """
    try:
        assert level in ('xcoarse', 'coarse', 'medium', 'fine', 'xfine')
    except AssertionError:
        raise ValueError("Mesh resolution level '{:s}' not recognised.".format(level))

    # Get number of turbines
    locs = op.region_of_interest
    n = len(locs)
    assert n >= 0

    # Get turbine diameter
    d = locs[0][2]
    for i in range(1, n):
        try:
            assert locs[i][2] == d
        except AssertionError:
            raise NotImplementedError("Turbines of different diameter not considered.")
    D = 2*d

    # Create mesh file and define parameters
    f = open(os.path.join(filepath, '{:s}_{:d}.geo'.format(level, op.offset)), 'w+')
    f.write('W={:.1f};     // width of channel\n'.format(op.domain_width))
    f.write('L={:.1f};      // length of channel\n'.format(op.domain_length))
    f.write('D={:.1f};     // turbine diameter\n'.format(D))
    f.write('dx1={:.1f};   // outer resolution\n'.format(op.resolution[level]['outer']))
    f.write('dx2={:.1f};   // inner resolution\n'.format(op.resolution[level]['inner']))
    for i in range(n):
        f.write('xt{:d}={:.1f};  // x-location of turbine {:d}\n'.format(i, locs[i][0], i+1))
        f.write('yt{:d}={:.1f};  // y-location of turbine {:d}\n'.format(i, locs[i][1], i+1))

    # Domain corners
    f.write('Point(1) = {0., 0., 0., dx1};\n')
    f.write('Point(2) = {L,  0., 0., dx1};\n')
    f.write('Point(3) = {L,  W,  0., dx1};\n')
    f.write('Point(4) = {0., W,  0., dx1};\n')

    # Turbine footprint corners
    for i in range(n):
        f.write('Point({:d}) = {{xt{:d}-D/2, yt{:d}-D/2, 0., dx2}};\n'.format(5+4*i, i, i))
        f.write('Point({:d}) = {{xt{:d}+D/2, yt{:d}-D/2, 0., dx2}};\n'.format(5+4*i+1, i, i))
        f.write('Point({:d}) = {{xt{:d}+D/2, yt{:d}+D/2, 0., dx2}};\n'.format(5+4*i+2, i, i))
        f.write('Point({:d}) = {{xt{:d}-D/2, yt{:d}+D/2, 0., dx2}};\n'.format(5+4*i+3, i, i))
    for j in range(1, 5):
        f.write('Line({:d}) = {{{:d}, {:d}}};\n'.format(j, j, j%4+1))
    for i in range(n):
        for j in range(4):
            f.write('Line({:d}) = {{{:d}, {:d}}};\n'.format(5+4*i+j, 5+4*i+j, 5+4*i+(j+1)%4))

    # Domain boundary tags
    f.write('Physical Line(1) = {4};   // Left boundary\n')
    f.write('Physical Line(2) = {2};   // Right boundary\n')
    f.write('Physical Line(3) = {1,3}; // Sides\n')

    # Domain boundary edges
    f.write('// outside loop\n')
    f.write('Line Loop(1) = {1, 2, 3, 4};\n')

    # Turbine footprint edges
    loop_str = 'Line Loop({:d}) = {{{:d}, {:d}, {:d}, {:d}}};\n'
    for i in range(n):
        f.write('// inside loop{:d}\n'.format(i))
        f.write(loop_str.format(i+2, 5+4*i, 5+4*i+1, 5+4*i+2, 5+4*i+3))

    # Plane surfaces
    plane_surface = 'Plane Surface(1) = {1, '
    for i in range(n-1):
        plane_surface += '{:d}, '.format(i+2)
    plane_surface += str(n+1) + '};\n'
    f.write(plane_surface)
    for i in range(n):
        f.write('Plane Surface({:d}) = {{{:d}}};\n'.format(i+2, i+2))

    # General domain surface ID
    f.write('// id outside turbine\n')
    f.write('Physical Surface(1) = {1};\n')

    # Surface IDs for individual turbines
    for i in range(n):
        f.write('// id inside turbine{:d}\n'.format(i+1))
        f.write('Physical Surface({:d}) = {{{:d}}};\n'.format(i+2, i+2))
    f.close()
