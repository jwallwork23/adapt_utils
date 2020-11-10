from firedrake import BoxMesh, Constant

from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


__all__ = ["PointDischarge3dOptions"]


# TODO: Analytical solution
# TODO: Calibrate radius

class PointDischarge3dOptions(PointDischarge2dOptions):
    r"""
    Parameters for a 3D extension of the 'Point source with diffusion' test case from TELEMAC-2D
    validation document version 7.0.

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi \;\mathrm dx,

    where :math:`A` is a spherical 'receiver' region.

    :kwarg approach: Mesh adaptation strategy,
    :kwarg offset: Shift in x-direction for source location.
    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, level=0, aligned=True, **kwargs):
        super(PointDischarge3dOptions, self).__init__(aligned=aligned, **kwargs)
        # self.qoi_quadrature_degree = 12
        self.qoi_quadrature_degree = 1

        # Simple 3D extension of 2D problem
        n = 2**level
        self.default_mesh = BoxMesh(100*n, 20*n, 20*n, 50, 10, 10)
        self.region_of_interest = [(20.0, 5.0, 5.0, 0.5)] if aligned else [(20.0, 7.5, 2.5, 0.5)]
        self.base_velocity = [1.0, 0.0, 0.0]

        # Adaptation parameters
        self.h_min = 1.0e-30
        self.h_max = 1.0e+06
        # self.hessian_recovery = 'parts'
        self.hessian_recovery = 'dL2'

        # Solver parameters
        self.solver_parameters['tracer'] = {
            'ksp_type': 'gmres',
            'pc_type': 'sor',
        }
        # self.solver_parameters['tracer'] = {
        #     'mat_type': 'aij',
        #     'ksp_type': 'preonly',
        #     'pc_type': 'lu',
        #     'pc_factor_mat_solver_type': 'mumps',
        # }

    def set_boundary_conditions(self, prob, i):
        zero = Constant(0.0)
        boundary_conditions = {
            'tracer': {
                1: {'value': zero},
                2: {},
                3: {'diff_flux': zero},
                4: {'diff_flux': zero},
                5: {'diff_flux': zero},
                6: {'diff_flux': zero},
            }
        }
        return boundary_conditions

    def set_calibrated_radius(self):
        r = 0.2  # TODO: Rerun for 3D case
        return [(1.0 + self.shift, 5.0, 5.0, r)]

    def analytical_solution(self, fs):
        raise NotImplementedError

    def analytical_qoi(self, mesh=None):
        raise NotImplementedError
