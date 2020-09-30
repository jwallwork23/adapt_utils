import argparse
import os

from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# Parse for refinement level
parser = argparse.ArgumentParser()
parser.add_argument("refinement_level", help="Number of refinements of farm region")
level = int(parser.parse_args().refinement_level)

# Boiler plate
code = "//" + 80*"*" + """
// This geometry file was automatically generated using the `meshgen.py` script
// with refinement level {:d}.
""".format(level) + "//" + 80*"*" + "\n\n"

# Domain and turbine specification
op = TurbineArrayOptions(1.0)
code += "// Domain and turbine specification\n"
code += "L = {:.0f};\n".format(op.domain_length)
code += "W = {:.0f};\n".format(op.domain_width)
code += "D = {:.0f};\n".format(op.turbine_diameter)
code += "d = {:.0f};\n".format(op.turbine_width)
code += "deltax = 10*D;\ndeltay = 7.5*D;\n"
code += "dx = 100;\n"
dxfarm = 24*0.5**level
code += "dxfarm = {:.0f};\n".format(dxfarm)
code += "dxturbine = {:.0f};\n".format(min(dxfarm, 6))

# Channel geometry
code += """
// Channel
Point(1) = {-L/2, -W/2, 0, dx};
Point(2) = {L/2, -W/2, 0, dx};
Point(3) = {L/2, W/2, 0, dx};
Point(4) = {-L/2, W/2, 0, dx};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Line Loop(1) = {1, 2, 3, 4};
"""

signs = ([-1, 1, 1, -1], [-1, -1, 1, 1])

# Code snippets
point_str = "Point(%d) = {%d*d/2 + %d*deltax, %d*D/2 + %d*deltay, 0, dxturbine};\n"
line_str = "Line(%d) = {%d, %d};\n"
loop_str = "Line Loop(%d) = {%d, %d, %d, %d};\n"

# Turbines
point = 5
line = 5
loop = 2
for col in range(5):
    for row in range(3):
        tag = op.array_ids[row][col]
        code += "\n// turbine %d\n" % (loop-1)
        for s1, s2 in zip(*signs):
            code += point_str % (point, s1, -2+col, s2, 1-row)
            point += 1
        for i in range(4):
            code += line_str % (line+i, line+i, line+((i+1) % 4))
        code += loop_str % (loop, line, line+1, line+2, line+3)
        line += 4
        loop += 1

# Refined region around turbines
code += "\n// Refined region around the turbines\n"
point_str = "Point(%d) = {%d*3*deltax, %d*1.3*deltay, 0, dxfarm};\n"
for s1, s2 in zip(*signs):
    code += point_str % (point, s1, s2)
    point += 1
for i in range(4):
    code += line_str % (line+i, line+i, line+((i+1) % 4))
code += loop_str % (loop, line, line+1, line+2, line+3)
line += 4

# Surfaces
code += """
// Surfaces
Plane Surface(1) = {1, 17};
"""
surface_str = "Plane Surface(%d) = {%d};\n"
for surface in range(2, 17):
    code += surface_str % (surface, surface)
code += "Plane Surface(17) = {17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};\n"

# Physical surfaces
code += "\n// Physical surfaces\nPhysical Surface(1) = {1, 17};"
surface_str = "\nPhysical Surface(%d) = {%d};"
for surface in range(2, 17):
    code += surface_str % (surface, surface)

# Write to file
with open(os.path.join(op.resource_dir, "channel_box_{:d}.geo".format(level)), 'w+') as f:
    f.write(code)
