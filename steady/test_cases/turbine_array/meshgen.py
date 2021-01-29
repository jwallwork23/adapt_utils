import argparse
import os

from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions


# Parse for refinement level and offset
parser = argparse.ArgumentParser()
parser.add_argument('refinement_level', help='Mesh resolution level.')
parser.add_argument('offset', help='Offset in the vertical in number of turbine diameters.')
parser.add_argument('-dx_refined', help='Mesh resolution for refined region')
args = parser.parse_args()
level = int(args.refinement_level)
offset = int(args.offset)

# Boiler plate
code = "//" + 80*"*" + """
// This geometry file was automatically generated using the `meshgen.py` script
// with refinement level {:d}.
""".format(level) + "//" + 80*"*" + "\n\n"

# Domain and turbine specification
op = TurbineArrayOptions(level=level, offset=offset, meshgen=True)
dx_refined = None if args.dx_refined is None else op.base_inner_res*0.5**level
locs = op.region_of_interest
n = len(locs)
code += "// Domain and turbine specification\n"
code += "L = {:.1f};\n".format(op.domain_length)
code += "W = {:.1f};\n".format(op.domain_width)
code += "D = {:.1f};\n".format(op.turbine_diameter)
dx_outer = op.base_outer_res
dx_inner = op.base_inner_res*0.5**level
if dx_refined is None:
    dx_outer *= 0.5**level
code += "dx_outer = {:.1f};\n".format(dx_outer)
code += "dx_inner = {:.1f};\n".format(dx_inner)
if dx_refined is not None:
    assert offset == 0
    code += "dx_refined = {:.1f};\n".format(dx_refined)
for i in range(n):
    code += "xt{:d}={:.1f};  // x-location of turbine {:d}\n".format(i, locs[i][0], i+1)
    code += "yt{:d}={:.1f};  // y-location of turbine {:d}\n".format(i, locs[i][1], i+1)

# Counters
line = 1
point = 1

# Domain
code += """
// Domain and turbine footprints
Point(1) = {0, 0, 0, dx_outer};
Point(2) = {L, 0, 0, dx_outer};
Point(3) = {L, W, 0, dx_outer};
Point(4) = {0, W, 0, dx_outer};
"""
point += 4
for j in range(1, 5):
    code += "Line({:d}) = {{{:d}, {:d}}};".format(line, j, j % 4+1)
    if j < 4:
        code += '\n'
    line += 1
code += """
Physical Line(1) = {4};   // Left boundary
Physical Line(2) = {2};   // Right boundary
Physical Line(3) = {1,3}; // Sides
Line Loop(1) = {1, 2, 3, 4};  // outside loop
"""

# Turbine footprints
for i in range(n):
    code += "Point({:d}) = {{xt{:d}-D/2, yt{:d}-D/2, 0., dx_inner}};\n".format(point, i, i)
    code += "Point({:d}) = {{xt{:d}+D/2, yt{:d}-D/2, 0., dx_inner}};\n".format(point+1, i, i)
    code += "Point({:d}) = {{xt{:d}+D/2, yt{:d}+D/2, 0., dx_inner}};\n".format(point+2, i, i)
    code += "Point({:d}) = {{xt{:d}-D/2, yt{:d}+D/2, 0., dx_inner}};\n".format(point+3, i, i)
    point += 4
for i in range(n):
    for j in range(4):
        code += "Line({:d}) = {{{:d}, {:d}}};\n".format(line, 5+4*i+j, 5+4*i+(j+1) % 4)
        line += 1

loop = 2
loop_str = "Line Loop({:d}) = {{{:d}, {:d}, {:d}, {:d}}};  // {:s}\n"
for i in range(n):
    label = "inside loop {:d}".format(i+1)
    code += loop_str.format(loop, 5+4*i, 5+4*i+1, 5+4*i+2, 5+4*i+3, label)
    loop += 1

# Refined region
if dx_refined is not None:
    code += """
// Refined region
Point({:d}) = {{400, 200, 0, dx_refined}};
Point({:d}) = {{1000, 200, 0, dx_refined}};
Point({:d}) = {{1000, 300, 0, dx_refined}};
Point({:d}) = {{400, 300, 0, dx_refined}};
""".format(point, point+1, point+2, point+3)
    for i in range(4):
        code += "Line({:d}) = {{{:d}, {:d}}};\n".format(line+i, point+i, point+((i+1) % 4))
    code += loop_str.format(loop, line, line+1, line+2, line+3, "box loop")

# Plane surfaces
plane_surface = "\n// Surfaces\nPlane Surface(1) = {1"
surface = 2
for i in range(n):
    plane_surface += ", {:d}".format(surface)
    surface += 1
if dx_refined is not None:
    plane_surface += ", {:d}".format(surface)
code += plane_surface + "};\n"
surface = 2
for i in range(n):
    code += "Plane Surface({:d}) = {{{:d}}};\n".format(surface, surface)
    surface += 1
if dx_refined is not None:
    plane_surface = "Plane Surface(%d) = {2" % surface
    for i in range(3, surface+1):
        plane_surface += ", {:d}".format(i)
    code += plane_surface + "};\n"
    code += "Plane Surface({:d}) = {{{:d}}};\n".format(surface+1, surface)

# Surface IDs
code += "Physical Surface(1) = {1};  // outside turbine\n"
surface = 2
for i in range(n):
    code += "Physical Surface({:d}) = {{{:d}}};  // inside turbine {:d}\n".format(surface, surface, i+1)
    surface += 1
if dx_refined is not None:
    code += "Physical Surface({:d}) = {{{:d}}};  // refined region\n".format(surface, surface)

# Write to file
if dx_refined is None:
    fname = op.mesh_file.replace('.msh', '.geo')
else:
    fname = os.path.join(op.mesh_dir, 'channel_refined_{:d}.geo'.format(level))
with open(fname, 'w+') as f:
    f.write(code)
