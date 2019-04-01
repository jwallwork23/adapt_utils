from thetis_adjoint import *
#from fenics_adjoint.solving import SolveBlock       # For extracting adjoint solutions
#from fenics_adjoint.projection import ProjectBlock  # Exclude projections from tape reading
import pyadjoint
import math
# op2.init(log_level=INFO)

from adapt_utils.turbine.options import TurbineOptions
from adapt_utils.solver import BaseProblem


__all__ = ["SteadyTurbineProblem"]


class SteadyTurbineProblem(BaseProblem):
    """
    TODO: documentation
    """
    def __init__(self,
                 mesh=RectangleMesh(100, 20, 1000., 200.),
                 finite_element=VectorElement("DG", triangle, 1)*FiniteElement("DG", triangle, 1),
                 approach='fixed_mesh',
                 stab=None,  # TODO
                 discrete_adjoint=True,
                 op=TurbineOptions(),
                 high_order=False):  # TODO
        super(SteadyTurbineProblem, self).__init__(mesh,
                                                   finite_element,
                                                   approach,
                                                   stab,
                                                   True,
                                                   discrete_adjoint,
                                                   op,
                                                   high_order)

        # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output
        # to 7 digits) in roughly 800 timesteps of 20s with SteadyState we only do 1 timestep 
        # (t_end should be slightly smaller than timestep to achieve this)
        self.t_end = 0.9*op.dt

        # physical fields and boundary values
        depth = 40.
        self.bathy = Constant(depth)
        self.viscosity = Constant(self.op.viscosity)
        self.drag_coefficient = Constant(self.op.drag_coefficient)
        self.inflow = Function(self.P1_vec).interpolate(as_vector([3., 0.]))

        # correction to account for the fact that the thrust coefficient is based on an upstream
        # velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        # Piggott 2016, eq. (15))
        D = self.op.turbine_diameter
        A_T = math.pi*(D/2)**2
        self.correction = 4/(1+math.sqrt(1-A_T/(depth*D)))**2
        self.op.thrust_coefficient *= self.correction
        # NOTE, that we're not yet correcting power output here, so that will be overestimated

        # parameters for adjoint computation
        self.gradient_field = self.bathy
        z, zeta = self.adjoint_solution.split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")

    def setup_equation(self):
        """
        Create a Thetis FlowSolver2d object for solving the (modified) shallow water equations.
        """
        self.solver_obj = solver2d.FlowSolver2d(self.mesh, self.bathy)
        options = self.solver_obj.options
        op = self.op
        options.timestep = op.dt
        options.simulation_export_time = op.dt
        options.simulation_end_time = self.t_end
        options.output_directory = op.directory()
        options.check_volume_conservation_2d = True
        options.use_grad_div_viscosity_term = op.symmetric_viscosity
        options.element_family = op.family
        options.timestepper_type = 'SteadyState'
        options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
        options.timestepper_options.solver_parameters['snes_monitor'] = None
        #options.timestepper_options.implicitness_theta = 1.0
        options.horizontal_viscosity = self.viscosity
        options.quadratic_drag_coefficient = self.drag_coefficient
        options.use_lax_friedrichs_velocity = False  # TODO
        options.use_grad_depth_viscosity_term = False

        # assign boundary conditions
        left_tag = 1
        right_tag = 2
        top_bottom_tag = 3
        freeslip_bc = {'un': Constant(0.)}
        self.solver_obj.bnd_functions['shallow_water'] = {
          left_tag: {'uv': self.inflow},
          right_tag: {'elev': Constant(0.)},
          top_bottom_tag: freeslip_bc,
        }

        # we haven't meshed the turbines with separate ids, so define a farm everywhere
        # and make it have a density of 1/D^2 inside the two DxD squares where the turbines are
        # and 0 outside
        scaling = len(op.region_of_interest)/assemble(op.bump(self.mesh)*dx)
        turbine_density = op.bump(self.mesh, scale=scaling)
        #File(op.directory()+'Bump.pvd').write(turbine_density)

        farm_options = TidalTurbineFarmOptions()
        farm_options.turbine_density = turbine_density
        farm_options.turbine_options.diameter = op.turbine_diameter
        farm_options.turbine_options.thrust_coefficient = op.thrust_coefficient
        # turbine drag is applied everywhere (where the turbine density isn't zero)
        options.tidal_turbine_farms["everywhere"] = farm_options

        # callback that computes average power
        self.callback = turbines.TurbineFunctionalCallback(self.solver_obj)
        self.solver_obj.add_callback(self.callback, 'timestep')

        self.solver_obj.assign_initial_conditions(uv=as_vector((3.0, 0.0)))

    def solve(self):
        if not hasattr(self, 'solver_obj'):
            self.setup_equation()
        self.solver_obj.iterate()
        self.solution = self.solver_obj.fields.solution_2d

    def objective_functional(self):
        return self.callback.average_power

    # TODO: Error estimators
