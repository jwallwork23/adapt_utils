from adapt_utils.turbine.options import *
import os


__all__ = ["generate_geo_file"]


def generate_geo_file(op, level='coarse', offset=False, filepath='.'):
    try:
        assert level in ('xcoarse', 'coarse', 'medium', 'fine', 'xfine')
    except:
        raise NotImplementedError
    locs = op.region_of_interest
    n = len(locs)
    assert n > 0
    label = '{:s}_{:d}'.format(level, n)
    if offset:
        label += '_offset'
    d = locs[0][2]
    for i in range(1, n):
        assert locs[i][2] == d
    D = 2*d
    f = open(os.path.join(filepath, label + '_turbine.geo'), 'w+')
    if n < 3:
        f.write('W=200.;     // width of channel\n')
        f.write('L=1e3;      // length of channel\n')
        if level == 'xcoarse':
            dx1 = 40.
            dx2 = 8.
        elif level == 'coarse':
            dx1 = 20.
            dx2 = 4.
        elif level == 'medium':
            dx1 = 10.
            dx2 = 2.
        elif level == 'fine':
            dx1 = 5.
            dx2 = 1.
        else:
            dx1 = 2.5
            dx2 = 0.5
    elif n == 15:
        f.write('W=1e3.;     // width of channel\n')
        f.write('L=3e3;      // length of channel\n')
        raise NotImplementedError
    else:
        raise NotImplementedError
    f.write('D=%.1f;     // turbine diameter\n' % D)
    f.write('dx1=%.1f;   // outer resolution\n' % dx1)
    f.write('dx2=%.1f;   // inner resolution\n' % dx2)
    for i in range(n):
        f.write('xt%d=%.1f;  // x-location of turbine %d\n' % (i, locs[i][0], i+1))
        f.write('yt%d=%.1f;  // y-location of turbine %d\n' % (i, locs[i][1], i+1))
    f.write('Point(1) = {0., 0., 0., dx1};\n')
    f.write('Point(2) = {L,  0., 0., dx1};\n')
    f.write('Point(3) = {L,  W,  0., dx1};\n')
    f.write('Point(4) = {0., W,  0., dx1};\n')
    for i in range(n):
        f.write('Point(%d) = {xt%d-D/2, yt%d-D/2, 0., dx2};\n' % (5+4*i, i, i))
        f.write('Point(%d) = {xt%d+D/2, yt%d-D/2, 0., dx2};\n' % (5+4*i+1, i, i))
        f.write('Point(%d) = {xt%d+D/2, yt%d+D/2, 0., dx2};\n' % (5+4*i+2, i, i))
        f.write('Point(%d) = {xt%d-D/2, yt%d+D/2, 0., dx2};\n' % (5+4*i+3, i, i))
    f.write('Line(1) = {1, 2};\n')
    f.write('Line(2) = {2, 3};\n')
    f.write('Line(3) = {3, 4};\n')
    f.write('Line(4) = {4, 1};\n')
    for i in range(n):
        f.write('Line(%d) = {%d, %d};\n' % (5+4*i, 5+4*i, 5+4*i+1))
        f.write('Line(%d) = {%d, %d};\n' % (5+4*i+1, 5+4*i+1, 5+4*i+2))
        f.write('Line(%d) = {%d, %d};\n' % (5+4*i+2, 5+4*i+2, 5+4*i+3))
        f.write('Line(%d) = {%d, %d};\n' % (5+4*i+3, 5+4*i+3, 5+4*i))
    f.write('Physical Line(1) = {4};   // Left boundary\n')
    f.write('Physical Line(2) = {2};   // Right boundary\n')
    f.write('Physical Line(3) = {1,3}; // Sides\n')
    f.write('// outside loop\n')
    f.write('Line Loop(1) = {1, 2, 3, 4};\n')
    for i in range(n):
        f.write('// inside loop%d\n' % i)
        f.write('Line Loop(%d) = {%d, %d, %d, %d};\n' % (i+2, 5+4*i, 5+4*i+1, 5+4*i+2, 5+4*i+3))
    plane_surface = 'Plane Surface(1) = {1, '
    for i in range(n-1):
        plane_surface += '%d, ' % (i+2)
    plane_surface += '%d};\n' % (n+1)
    f.write(plane_surface)
    for i in range(n):
        f.write('Plane Surface(%d) = {%d};\n' % (i+2, i+2))
    f.write('// id outside turbine\n')
    f.write('Physical Surface(1) = {1};\n')
    for i in range(n):
        f.write('// id inside turbine%d\n' % (i+1))
        f.write('Physical Surface(%d) = {%d};\n' % (i+2, i+2))
    f.close()
