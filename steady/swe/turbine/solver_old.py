from thetis import *

from adapt_utils.steady.swe.solver import *


__all__ = ["SteadyTurbineProblem"]


class SteadyTurbineProblem(SteadyShallowWaterProblem):
    """
    General solver object for stationary tidal turbine problems.
    """
    def extra_setup(self):
        """
        We haven't meshed the turbines with separate ids, so define a farm everywhere and make it
        have a density of 1/D^2 inside the DxD squares where the turbines are and 0 outside.
        """
        op = self.op
        num_turbines = len(op.region_of_interest)
        scaling = num_turbines/assemble(op.bump(self.mesh)*dx)
        self.turbine_density = op.bump(self.mesh, scale=scaling)
        # scaling = num_turbines/assemble(op.box(self.P0)*dx)
        # self.turbine_density = op.box(self.P0, scale=scaling)
        self.farm_options = TidalTurbineFarmOptions()
        self.farm_options.turbine_density = self.turbine_density
        self.farm_options.turbine_options.diameter = op.turbine_diameter
        self.farm_options.turbine_options.thrust_coefficient = op.thrust_coefficient

        A_T = pi*(op.turbine_diameter/2.0)**2  # TODO: Why circular area?
        self.C_D = op.thrust_coefficient*A_T*self.turbine_density/2.0

        # Turbine drag is applied everywhere (where the turbine density isn't zero)
        self.solver_obj.options.tidal_turbine_farms["everywhere"] = self.farm_options

        # Callback that computes average power
        self.cb = turbines.TurbineFunctionalCallback(self.solver_obj)
        self.solver_obj.add_callback(self.cb, 'timestep')

    def extra_residual_terms(self):
        u, eta = self.solution.split()
        z, zeta = self.adjoint_solution.split()
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*inner(z, u)/H

    def extra_strong_residual_terms_momentum(self):
        u, eta = self.solution.split()
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*u/H

    def get_qoi_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)

    def quantity_of_interest(self):
        self.qoi = self.cb.average_power
        return self.qoi

    def quantity_of_interest_form(self):
        return self.C_D*pow(inner(split(self.solution)[0], split(self.solution)[0]), 1.5)*dx