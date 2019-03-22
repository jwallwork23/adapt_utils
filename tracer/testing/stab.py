from firedrake import *
from time import clock
#from tracer.options import *
from tracer.solver import *


# No stabilisation
tp1 = TracerProblem()
tp1.setup_equation()
t_no_stab = clock()
tp1.solve()
t_no_stab = clock() - t_no_stab

# SU stabilisation
tp2 = TracerProblem(stab='SU')
tp2.setup_equation()
t_su = clock()
tp2.solve()
t_su = clock() - t_su

# SUPG stabilisation
tp3 = TracerProblem(stab='SUPG')
tp3.setup_equation()
t_supg = clock()
tp3.solve()
t_supg = clock() - t_supg

# Print results
print("{:20s} {:8s}".format("Stabilisation method", "Time (s)"))
print("{:20s} {:8.2f}".format("None", t_no_stab))
print("{:20s} {:8.2f}".format("SU", t_su))
print("{:20s} {:8.2f}".format("SUPG", t_supg))
