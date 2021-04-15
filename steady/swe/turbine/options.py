from thetis import *
from thetis.configuration import *

<<<<<<< HEAD
<<<<<<<< HEAD:steady/swe/turbine/options.py
from adapt_utils.unsteady.options import CoupledOptions  # TODO: Don't refer to unsteady
========
from adapt_utils.options import CoupledOptions
>>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a:swe/turbine/options.py
=======
from adapt_utils.swe.turbine.options import TurbineOptions
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


__all__ = ["TurbineOptions"]


<<<<<<< HEAD
<<<<<<<< HEAD:steady/swe/turbine/options.py
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
# Default: Newton with line search; solve linear system exactly with LU factorisation
lu_params = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
<<<<<<< HEAD
    # 'snes_rtol': 1e-3,
    'snes_rtol': 1e-8,
    # 'snes_atol': 1e-16,
=======
    'snes_rtol': 1e-8,
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    'snes_max_it': 100,
    'snes_linesearch_type': 'bt',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_type': 'preonly',
    'ksp_converged_reason': None,
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}
<<<<<<< HEAD
# TODO: 'Physics based' fieldsplit approach
default_params = lu_params
keys = {key for key in default_params if 'snes' not in key}
default_adjoint_params = {}
default_adjoint_params.update(default_params)


========
>>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a:swe/turbine/options.py
class TurbineOptions(CoupledOptions):
    """
    Parameter class for tidal turbine problems solved using a drag parametrisation for the turbines
    within a depth-averaged shallow water model.

    Physical ID tags for the turbines can be provided using :attr:`farm_ids`. By default, a single
    ID 'everywhere' is used which does not distinguish between separate turbines.
    """

<<<<<<<< HEAD:steady/swe/turbine/options.py
    # Turbine parametrisation
    turbine_diameter = PositiveFloat(18.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(0.8).tag(config=True)

    def __init__(self, num_iterations=1, bathymetry_space=None, timestepper='SteadyState', **kwargs):
        self.base_bathymetry = 40.0
        self.set_bathymetry(bathymetry_space)
        super(TurbineOptions, self).__init__(**kwargs)
        self.solve_swe = True
        self.solve_tracer = False
        self.timestepper = timestepper
        self.dt = 20.0
        self.end_time = num_iterations*self.dt - 0.2
        self.lax_friedrichs_velocity_scaling_factor = Constant(1.0)

        # Solver parameters
        self.solver_parameters = default_params
        self.adjoint_solver_parameters = default_adjoint_params

        # Adaptivity
        self.h_min = 1e-5
        self.h_max = 500.0

    def max_depth(self):
        """Compute maximum depth from bathymetry field."""
        if hasattr(self, 'bathymetry'):
            if isinstance(self.bathymetry, Constant):
                return self.bathymetry.values()[0]
            elif isinstance(self.bathymetry, Function):
                return self.bathymetry.vector().gather().max()
            else:
                raise ValueError("Bathymetry format cannot be understood.")
        else:
            assert hasattr(self, 'base_bathymetry')
            return self.base_bathymetry

    def set_bathymetry(self, fs):
        return Constant(self.base_bathymetry)

    def set_quadratic_drag_coefficient(self, fs):
        return Constant(0.0025)
========
    # Turbine parameters
    turbine_diameter = PositiveFloat(18.0, help=r"""
        Diameter of the circular region swept in the vertical by the turbine blades.

        The 'swept area' of the turbine is calculated as :math:`pi*\frac{D^2}4`, where
        :math:`D` is the turbine diameter.

        The 'footprint area' of the turbine is calculated as :math:`D*W`, where :math:`W`
        is the 'width' of the turbine footprint. By default, this is set to the diameter.
        However, in some cases it is appropriate to choose a smaller value. This is the case
        if the flow is unidirectional, for example.
        """).tag(config=False)
    turbine_width = PositiveFloat(None, allow_none=True, help="""
        Width of the rectangular turbine footprint region covered in the horizontal.
        """).tag(config=False)
    thrust_coefficient = NonNegativeFloat(0.6).tag(config=True)

    # Physics
    sea_water_density = PositiveFloat(1030.0).tag(config=True)

    # --- Setup

    def __init__(self, **kwargs):
        super(TurbineOptions, self).__init__(**kwargs)
        self.farm_ids = ["everywhere"]
        if self.turbine_width is None:
            self.turbine_width = self.turbine_diameter
        else:
            assert self.turbine_width <= self.turbine_diameter

        # Timestepping
        self.timestepper = 'CrankNicolson'

        # Boundary forcing
        self.M2_tide_period = 12.4*3600.0
        self.T_tide = self.M2_tide_period
        self.dt_per_export = 10

    def set_quadratic_drag_coefficient(self, fs):
        """Constant background (quadratic) drag is set using :attr:`friction_coeff`."""
        return Constant(self.friction_coeff)
>>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a:swe/turbine/options.py

    def set_manning_drag_coefficient(self, fs):
        """We do not use the Manning friction formulation for tidal turbine modelling."""
        return

    def get_thrust_coefficient(self, correction=True):
        """
        Correction to account for the fact that the thrust coefficient is based on an upstream
        velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        Piggott 2016, eq. (15))

        NOTE: We're not yet correcting power output here, so that will be overestimated
        """
<<<<<<<< HEAD:steady/swe/turbine/options.py
        D = self.turbine_diameter
        A_T = pi*(D/2)**2
        correction = 4/(1+sqrt(1-A_T/(self.max_depth()*D)))**2
        self.thrust_coefficient *= correction
        # NOTE: We're not yet correcting power output here, so that will be overestimated
========
        if not correction:
            return self.thrust_coefficient
        self.print_debug("TURBINE: Computing thrust coefficient correction...")

        # Turbine geometry
        D = self.turbine_diameter
        W = self.turbine_width
        if not hasattr(self, 'max_depth'):  # TODO: Use a property
            self.get_max_depth()
        H = self.max_depth
        swept_area = pi*(D/2)**2    # area swept by turbine blades in the vertical
        footprint_area = D*W        # area of one turbine footprint
        cross_sectional_area = H*D  # cross-sectional area of water depth

        # Thrust correction
        thrust = self.thrust_coefficient
        drag = 0.5*thrust*swept_area/footprint_area
        correction = 4/(1 + sqrt(1 - thrust*swept_area/cross_sectional_area))**2
        thrust_corrected = thrust*correction
        drag_corrected = 0.5*thrust_corrected*swept_area/footprint_area

        # Debugging
        self.print_debug("TURBINE: Input thrust coefficient     = {:.2f}".format(thrust))
        self.print_debug("TURBINE: Input drag coefficient       = {:.2f}".format(drag))
        self.print_debug("TURBINE: Thrust correction            = {:.2f}".format(correction))
        self.print_debug("TURBINE: Corrected thrust coefficient = {:.2f}".format(thrust_corrected))
        self.print_debug("TURBINE: Corrected drag coefficient   = {:.2f}".format(drag_corrected))
        return thrust_corrected

    def get_max_depth(self, bathymetry=None):
        """
        Compute maximum depth from a provided bathymetry field. If no field is available then
        :attr:`base_bathymetry` is used instead.
        """
        if bathymetry is not None:
            if isinstance(bathymetry, Constant):
                self.max_depth = bathymetry.values()[0]
            elif isinstance(bathymetry, Function):
                self.max_depth = bathymetry.vector().gather().max()
            else:
                raise ValueError("Bathymetry format cannot be understood.")
        else:
            assert hasattr(self, 'base_bathymetry')
            self.max_depth = self.base_bathymetry

    # --- Tidal forcing

    def extract_data(self):
        """
        Extract tidal forcing time and elevation data from file as NumPy arrays.

        Note that this isn't *raw* data because it has been converted to appropriate units using
        `preproc.py`.
        """
        data_file = os.path.join(self.resource_dir, 'forcing.dat')
        if not os.path.exists(data_file):
            raise IOError("Tidal forcing data cannot be found in {:}.".format(self.resource_dir))
        times, data = [], []
        with open(data_file, 'r') as f:
            for line in f:
                time, dat = line.split()
                times.append(float(time))
                data.append(float(dat))
        return np.array(times), np.array(data)

    def interpolate_tidal_forcing(self):
        """
        Read tidal forcing data from the 'forcing.dat' file in the resource directory using the
        method :attr:`extract_data` and create a 1D linear interpolator.

        As a side-product, we determine the maximum amplitude of the tidal forcing and also the
        time period within which these data are available.
        """
        import scipy.interpolate as si

        times, data = self.extract_data()
        self.tidal_forcing_interpolator = si.interp1d(times, data)
        self.max_amplitude = np.max(np.abs(data))
        self.tidal_forcing_end_time = times[-1]

    # --- Power output

    def set_qoi_kernel(self, prob, i):
        r"""
        Assuming the QoI is power output,

      ..math::
            J(\mathbf q) = \int_\Omega C_t \|\mathbf u\|^3 \;\mathrm dx,

        this can be represented as an inner product with :math:`mathbf q`
        using the kernel function

      ..math::
            \mathbf k(\mathbf q) = (\frac13 C_t \|\mathbf u\|\mathbf u, 0).

        That is,

      ..math::
            J(\mathbf q) = \int_\Omega \mathbf k(\mathbf q) \cdot \mathbf q \;\mathrm dx.
        """
        prob.kernels[i] = Function(prob.V[i])
        u, eta = prob.fwd_solutions[i].split()
        k_u, k_eta = prob.kernels[i].split()
        # k_u.interpolate(Constant(1/3)*prob.turbine_densities[i]*sqrt(inner(u, u))*u)
        k_u.interpolate(prob.turbine_densities[i]*sqrt(inner(u, u))*u)
>>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a:swe/turbine/options.py
=======
default_params = lu_params


class SteadyTurbineOptions(TurbineOptions):
    """
    Base class holding parameters for steady state tidal turbine problems.
    """

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.0).tag(config=True)
    thrust_coefficient = NonNegativeFloat(0.8).tag(config=True)

    # --- Setup

    def __init__(self, num_iterations=1, **kwargs):
        super(SteadyTurbineOptions, self).__init__(**kwargs)

        # Timestepping
        self.timestepper = 'SteadyState'
        self.dt = 20.0
        self.dt_per_export = 1
        self.end_time = num_iterations*self.dt - 0.2

        # Solver parameters
        self.solver_parameters = {'shallow_water': default_params}
        self.adjoint_solver_parameters = {'shallow_water': default_params}
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
