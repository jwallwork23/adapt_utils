import argparse
import os

from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions


# Parse for refinement level and offset
parser = argparse.ArgumentParser()
parser.add_argument('refinement_level', help='Mesh resolution level.')
parser.add_argument('offset', help='Offset in the vertical in number of turbine diameters.')
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
locs = op.region_of_interest
n = len(locs)
code += "// Domain and turbine specification\n"
code += "L = {:.1f};\n".format(op.domain_length)
code += "W = {:.1f};\n".format(op.domain_width)
code += "D = {:.1f};\n".format(op.turbine_diameter)
code += "dx1 = {:.1f};\n".format(op.base_outer_res*0.5**level)
code += "dx2 = {:.1f};\n".format(op.base_inner_res*0.5**level)
for i in range(n):
    code += "xt{:d}={:.1f};  // x-location of turbine {:d}\n".format(i, locs[i][0], i+1)
    code += "yt{:d}={:.1f};  // y-location of turbine {:d}\n".format(i, locs[i][1], i+1)

# Domain corners
code += """
// Domain and turbine footprints
Point(1) = {0., 0., 0., dx1};
Point(2) = {L,  0., 0., dx1};
Point(3) = {L,  W,  0., dx1};
Point(4) = {0., W,  0., dx1};
"""

# Turbine footprints
for i in range(n):
    code += "Point({:d}) = {{xt{:d}-D/2, yt{:d}-D/2, 0., dx2}};\n".format(5+4*i, i, i)
    code += "Point({:d}) = {{xt{:d}+D/2, yt{:d}-D/2, 0., dx2}};\n".format(5+4*i+1, i, i)
    code += "Point({:d}) = {{xt{:d}+D/2, yt{:d}+D/2, 0., dx2}};\n".format(5+4*i+2, i, i)
    code += "Point({:d}) = {{xt{:d}-D/2, yt{:d}+D/2, 0., dx2}};\n".format(5+4*i+3, i, i)
for j in range(1, 5):
    code += "Line({:d}) = {{{:d}, {:d}}};\n".format(j, j, j % 4+1)
for i in range(n):
    for j in range(4):
        code += "Line({:d}) = {{{:d}, {:d}}};\n".format(5+4*i+j, 5+4*i+j, 5+4*i+(j+1) % 4)

# Domain boundaries
code += """
// Domain boundary
Physical Line(1) = {4};   // Left boundary
Physical Line(2) = {2};   // Right boundary
Physical Line(3) = {1,3}; // Sides

// outside loop
Line Loop(1) = {1, 2, 3, 4};
"""

# Turbine footprint edges
loop_str = """
// inside loop {:d}
Line Loop({:d}) = {{{:d}, {:d}, {:d}, {:d}}};
"""
for i in range(n):
    code += loop_str.format(i+1, i+2, 5+4*i, 5+4*i+1, 5+4*i+2, 5+4*i+3)

# Plane surfaces
plane_surface = "Plane Surface(1) = {1, "
for i in range(n-1):
    plane_surface += "{:d}, ".format(i+2)
plane_surface += str(n+1) + "};\n"
code += plane_surface
for i in range(n):
    code += "Plane Surface({:d}) = {{{:d}}};\n".format(i+2, i+2)

# Surface IDs
code += "Physical Surface(1) = {1};  // id outside turbine\n"
for i in range(n):
    code += "Physical Surface({:d}) = {{{:d}}};  // id inside turbine {:d}\n".format(i+2, i+2, i+1)

# Write to file
with open(op.mesh_file.replace('msh', 'geo'), 'w+') as f:
    f.write(code)
