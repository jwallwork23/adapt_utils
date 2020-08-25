from thetis import *
from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions


__all__ = ["TurbineOptions"]


class TurbineOptions(CoupledOptions):
    """
    Parameter class for tidal turbine problems solved using a drag parametrisation for the turbines
    within a depth-averaged shallow water model.

    Physical ID tags for the turbines can be provided using :attr:`farm_ids`. By default, a single
    ID 'everywhere' is used which does not distinguish between separate turbines.
    """

    # Turbine parameters
    turbine_length = PositiveFloat(18.0).tag(config=False)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)

    # --- Setup

    def __init__(self, **kwargs):
        super(TurbineOptions, self).__init__(**kwargs)
        self.farm_ids = ["everywhere"]

        # Timestepping
        self.timestepper = 'CrankNicolson'

        # Boundary forcing
        self.M2_tide_period = 12.4*3600.0
        self.T_tide = self.M2_tide_period
        self.dt_per_export = 10

    def set_quadratic_drag_coefficient(self, fs):
        """Constant background (quadratic) drag is set using :attr:`friction_coeff`."""
        return Constant(self.friction_coeff)

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
        if not correction:
            return self.thrust_coefficient
        if hasattr(self, 'turbine_diameter'):
            D = self.turbine_diameter
        else:
            D = max(self.turbine_length, self.turbine_width)
        A_T = pi*(D/2)**2
        correction = 4/(1 + sqrt(1 - A_T/(self.max_depth*D)))**2
        return self.thrust_coefficient*correction

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
