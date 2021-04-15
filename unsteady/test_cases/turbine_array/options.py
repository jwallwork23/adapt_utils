from thetis import *
from thetis.configuration import *

import os

<<<<<<< HEAD
from adapt_utils.unsteady.swe.turbine.options import TurbineOptions
=======
from adapt_utils.swe.turbine.options import TurbineOptions
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


__all__ = ["TurbineArrayOptions"]


class TurbineArrayOptions(TurbineOptions):
<<<<<<< HEAD
    """Parameters for the unsteady 15 turbine array test case."""

    # Turbine parameters
    turbine_length = PositiveFloat(20.0).tag(config=False)
    turbine_width = PositiveFloat(5.0).tag(config=False)
    array_length = PositiveInteger(5).tag(config=False)
    array_width = PositiveInteger(3).tag(config=False)

    # Domain specification
    mesh_file = os.path.join(os.path.dirname(__file__), 'channel.msh')
    domain_length = PositiveFloat(3000.0).tag(config=False)
    domain_width = PositiveFloat(1000.0).tag(config=False)

    def __init__(self, **kwargs):
        super(TurbineArrayOptions, self).__init__(**kwargs)

        # Domain and mesh
        if os.path.exists(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)
        else:
            raise OSError("Need to make mesh before initialising TurbineArrayOptions object.")

        # Physics
        self.base_viscosity = 3.0
        self.base_bathymetry = 50.0
        self.max_depth = 50.0
        self.friction_coeff = 0.0025

        # Timestepping
        self.dt = 3.0
        self.T_tide = 0.1*self.M2_tide_period
        self.T_ramp = 1.0*self.T_tide
        self.end_time = self.T_ramp + 2.0*self.T_tide
        self.dt_per_export = 12

        # Tidal farm
        D = self.turbine_length
        d = self.turbine_width
        self.turbine_diameter = max(D, d)
        deltax = 10.0*D
        deltay = 7.5*D
        self.region_of_interest = []
        self.num_turbines = self.array_length*self.array_width
        for i in range(-2, 3):
            for j in range(-1, 2):
                self.region_of_interest.append((i*deltax, j*deltay, D, d))
        assert len(self.region_of_interest) == self.num_turbines
        self.turbine_tags = list(range(2, 2+self.num_turbines))
=======
    """
    Parameters for the unsteady 15 turbine array test case from [Divett et al. 2013].
    """
    resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    # Turbine parameters
    turbine_diameter = PositiveFloat(20.0).tag(config=False)
    turbine_width = PositiveFloat(5.0).tag(config=False)
    array_length = PositiveInteger(5).tag(config=False)
    array_width = PositiveInteger(3).tag(config=False)
    num_turbines = PositiveInteger(15).tag(config=False)

    # Domain specification
    domain_length = PositiveFloat(3000.0).tag(config=False)
    domain_width = PositiveFloat(1000.0).tag(config=False)

    # Physics
    characteristic_velocity = FiredrakeVectorExpression(
        Constant(as_vector([1.5, 0.0]))
    ).tag(config=False)
    thrust_coefficient = NonNegativeFloat(2.985).tag(config=True)

    def __init__(self, base_viscosity, target_viscosity=None, level=0, spun=False, **kwargs):
        super(TurbineArrayOptions, self).__init__(**kwargs)
        self.array_ids = np.array([[2, 5, 8, 11, 14],
                                   [3, 6, 9, 12, 15],
                                   [4, 7, 10, 13, 16]])
        self.farm_ids = tuple(self.array_ids.reshape((self.num_turbines, )))

        # Domain and mesh
        self.mesh_file = os.path.join(self.resource_dir, 'channel_box_{:d}.msh'.format(level))
        if os.path.exists(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)
        else:
            import warnings
            warnings.warn("Need to make mesh before initialising TurbineArrayOptions object.")

        # Physics
        self.base_viscosity = base_viscosity
        self.target_viscosity = target_viscosity or base_viscosity
        self.sponge_x = 200
        self.sponge_y = 100
        self.base_bathymetry = 50.0
        self.max_depth = 50.0
        self.friction_coeff = 0.0025
        self.recover_vorticity = True

        # Timestepping
        self.dt = 2.232
        self.T_tide = 0.1*self.M2_tide_period
        self.T_ramp = 3.855*self.T_tide
        self.end_time = self.T_tide if spun else self.T_ramp
        self.dt_per_export = 10

        # Tidal farm
        W = self.turbine_width
        D = self.turbine_diameter
        deltax = 10.0*D
        deltay = 7.5*D
        self.region_of_interest = []
        for i in range(-2, 3):
            for j in range(1, -2, -1):
                self.region_of_interest.append((i*deltax, j*deltay, W, D))
        assert len(self.region_of_interest) == self.num_turbines
        self.turbine_tags = list(range(2, 2 + self.num_turbines))
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

        # Boundary forcing
        self.max_amplitude = 0.5
        self.omega = 2*pi/self.T_tide
        self.elev_in = [None for i in range(self.num_meshes)]
        self.elev_out = [None for i in range(self.num_meshes)]
<<<<<<< HEAD

        # Solver parameters and discretisation
        self.stabilisation = None
        # self.stabilisation = 'lax_friedrichs'
=======
        self.spun = spun

        # Discretisation
        self.stabilisation = 'lax_friedrichs'
        self.use_automatic_sipg_parameter = True
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = True
        self.family = 'dg-cg'

    def set_viscosity(self, fs):
<<<<<<< HEAD
        sponge = False
        if sponge:
            x, y = SpatialCoordinate(fs.mesh())
            xmax = 1000.0
            ramp = 0.5
            eps = 20.0
            return interpolate(self.base_viscosity + exp(ramp*(x - xmax + eps)), fs)
        else:
            return Constant(self.base_viscosity)
=======
        """
        Set the viscosity to be the :attr:`target_viscosity` in the tidal farm region and
        :attr:`base_viscosity` elsewhere.
        """
        nu = Function(fs, name="Horizontal viscosity")

        # Get box around tidal farm
        D = self.turbine_diameter
        delta_x = 3*10*D
        delta_y = 1.3*7.5*D

        # Base viscosity and minimum viscosity
        nu_tgt = self.target_viscosity
        nu_base = self.base_viscosity
        if np.isclose(nu_tgt, nu_base):
            nu.assign(nu_base)
            return nu

        # Distance functions
        mesh = fs.mesh()
        x, y = SpatialCoordinate(mesh)
        dist_x = (abs(x) - delta_x)/self.sponge_x
        dist_y = (abs(y) - delta_y)/self.sponge_y
        dist_r = sqrt(dist_x**2 + dist_y**2)

        # Define viscosity field with a sponge condition
        nu.interpolate(
            conditional(
                And(x > -delta_x, x < delta_x),
                conditional(
                    And(y > -delta_y, y < delta_y),
                    nu_tgt,
                    min_value(nu_tgt*(1 - dist_y) + nu_base*dist_y, nu_base),
                ),
                conditional(
                    And(y > -delta_y, y < delta_y),
                    min_value(nu_tgt*(1 - dist_x) + nu_base*dist_x, nu_base),
                    min_value(nu_tgt*(1 - dist_r) + nu_base*dist_r, nu_base),
                ),
            )
        )

        # Enforce maximum Reynolds number
        if hasattr(self, 'max_reynolds_number'):
            Re_h, Re_h_min, Re_h_max = self.check_mesh_reynolds_number(nu)
            target = self.max_reynolds_number
            if Re_h_max > target:
                nu_enforce = self.enforce_mesh_reynolds_number(fs, target)
                nu.interpolate(conditional(Re_h > target, nu_enforce, nu))

        return nu
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

    def set_boundary_conditions(self, prob, i):
        self.elev_in[i] = Function(prob.V[i].sub(1))
        self.elev_out[i] = Function(prob.V[i].sub(1))
        inflow_tag = 4
        outflow_tag = 2
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in[i]},
                outflow_tag: {'elev': self.elev_out[i]},
            }
        }
        return boundary_conditions

    def get_update_forcings(self, prob, i, **kwargs):
<<<<<<< HEAD
        tc = Constant(0.0)
        hmax = Constant(self.max_amplitude)

        def update_forcings(t):
            tc.assign(t)
            self.elev_in[i].assign(hmax*cos(self.omega*(tc - self.T_ramp)))
            self.elev_out[i].assign(hmax*cos(self.omega*(tc - self.T_ramp) + pi))
=======
        """
        Simple tidal forcing with frequency :attr:`omega` and amplitude :attr:`max_amplitude`.

        :arg prob: :class:`AdaptiveTurbineProblem` object.
        :arg i: mesh index.
        """
        tc = Constant(0.0)
        hmax = Constant(self.max_amplitude)
        offset = self.T_ramp if self.spun else 0.0

        def update_forcings(t):
            tc.assign(t + offset)
            self.elev_in[i].assign(hmax*cos(self.omega*tc))
            self.elev_out[i].assign(hmax*cos(self.omega*tc + pi))
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

        return update_forcings

    def set_initial_condition(self, prob):
<<<<<<< HEAD
        u, eta = prob.fwd_solutions[0].split()
        x, y = SpatialCoordinate(prob.meshes[0])
        u.interpolate(as_vector([1e-8, 0.0]))
        eta.interpolate(-x/self.domain_length)
=======
        """
        Specify elevation at the start of the spin-up period so that it satisfies the boundary
        forcing and set an arbitrary small velocity.

        :arg prob: :class:`AdaptiveTurbineProblem` object.
        """
        assert not self.spun
        u, eta = prob.fwd_solutions[0].split()
        x, y = SpatialCoordinate(prob.meshes[0])

        # Set an arbitrary, small, non-zero velocity which satisfies the free-slip conditions
        u.interpolate(as_vector([1e-8, 0.0]))

        # Set the initial surface so that it satisfies the forced boundary conditions
        X = 2*x/self.domain_length  # Non-dimensionalised x
        eta.interpolate(-self.max_amplitude*X)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
