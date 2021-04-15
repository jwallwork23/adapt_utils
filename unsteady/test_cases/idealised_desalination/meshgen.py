import argparse
import os

from adapt_utils.unsteady.test_cases.idealised_desalination.options import *


# Parse for refinement level
parser = argparse.ArgumentParser()
parser.add_argument("refinement_level", help="Number of refinements in central box")
level = int(parser.parse_args().refinement_level)

# Boiler plate
code = "//" + 80*"*" + """
// This geometry file was automatically generated using the `meshgen.py` script
// with refinement level {:d}.
""".format(level) + "//" + 80*"*" + "\n\n"

# Domain and turbine specification
op = IdealisedDesalinationOutfallOptions()
code += "// Domain specification\n"
code += "L = {:.0f};\n".format(op.domain_length)
code += "W = {:.0f};\n".format(op.domain_width)
code += "deltax = 1000;\ndeltay = 250;\n"
code += "dx = 100;\n"
dx_refined = [24, 12, 6, 5, 4, 3][level]
code += "dx_refined = {:.1f};\n".format(dx_refined)

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

// Refined region
Point(5) = {-1*deltax, -1*deltay, 0, dx_refined};
Point(6) = {1*deltax, -1*deltay, 0, dx_refined};
Point(7) = {1*deltax, 1*deltay, 0, dx_refined};
Point(8) = {-1*deltax, 1*deltay, 0, dx_refined};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line Loop(2) = {5, 6, 7, 8};

// Surfaces
Plane Surface(1) = {1, 2};
Plane Surface(2) = {2};

// Physical surfaces
Physical Surface(1) = {1};
Physical Surface(2) = {2};
"""

# Write to file
with open(os.path.join(op.resource_dir, "channel_{:d}.geo".format(level)), 'w+') as f:
    f.write(code)
