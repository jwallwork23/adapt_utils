from __future__ import absolute_import
from thetis.utility import *
from .equation import Equation


class GOErrorEstimatorTerm(object):
    """
    Implements the component of a goal-oriented error estimator from a single term of the underlying
    equation.
    """
    def __init__(self, mesh):
        """
        :arg function_space: the :class:`FunctionSpace` the solution belongs to
        """
        self.P0 = FunctionSpace(mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)

    def restrict(self, arg):
        """
        Restrict a discontinuous object to an element.
        """
        return jump(arg, self.p0test)

    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        """
        Returns an UFL form of the dx terms.

        :arg arg: argument :class:`.Function` to take inner product with.
        :arg arg_old: a time lagged solution :class:`.Function`
        """
        raise NotImplementedError('Must be implemented in the derived class')

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        """
        Returns an UFL form of the dS terms.

        :arg arg: argument :class:`.Function` to take inner product with.
        :arg arg_old: a time lagged solution :class:`.Function`
        """
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        """
        Returns an UFL form of the ds terms.

        :arg arg: argument :class:`.Function` to take inner product with.
        :arg arg_old: a time lagged solution :class:`.Function`
        """
        return 0


class GOErrorEstimator(Equation):
    """
    Implements a goal-oriented error estimator, comprised of the corresponding terms from the
    underlying equation.
    """
    def __init__(self, *args, **kwargs):
        super(GOErrorEstimator, self).__init__(*args, **kwargs)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)

    def mass_term(self, solution, arg, vector=False, **kwargs):
        """
        Returns an UFL form of the solution weighted by the argument.

        :arg arg: argument :class:`.Function` to take inner product with.
        """
        mass = self.p0test*inner(solution, arg)*dx
        if vector:
            import numpy as np
            mass = np.array([mass])
        return mass

    def _create_element_residual(self, label, *args):
        self.residual_terms = 0
        for term in self.select_terms(label):
            self.residual_terms += term.element_residual(*args)
        self.residual = Function(self.P0, name="Element residual")

    def _create_inter_element_flux(self, label, *args):
        self.inter_element_flux_terms = 0
        for term in self.select_terms(label):
            self.inter_element_flux_terms += term.inter_element_flux(*args)
        self.flux = Function(self.P0, name="Inter-element flux terms")

    def _create_boundary_flux(self, label, *args):
        self.bnd_flux_terms = 0
        for term in self.select_terms(label):
            self.bnd_flux_terms += term.boundary_flux(*args)
        self.bnd = Function(self.P0, name="Boundary flux terms")

    def setup_components(self, *args):
        """
        Set up dx, dS and ds components of the error estimator as element-wise indicator functions.
        """
        self._create_element_residual(*args[:-1])
        self._create_inter_element_flux(*args[:-1])
        self._create_boundary_flux(*args)

    def element_residual(self):
        """
        Evaluate contribution of dx terms to the error estimator as element-wise indicator functions.
        """
        self.residual.assign(assemble(self.residual_terms))
        return self.residual

    def inter_element_flux(self):
        """
        Evaluate contribution of dS terms to the error estimator as element-wise indicator functions.

        NOTE: The mass matrix is diagonal in P0 space so applying a Jacobi PC is an exact solve!
        """
        if self.inter_element_flux_terms == 0:
            print_output("NOTE: No inter-element flux terms detected.")
            self.flux.assign(0.0)
        else:
            mass_term = self.p0test*self.p0trial*dx
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
            solve(mass_term == self.inter_element_flux_terms, self.flux, solver_parameters=params)
        return self.flux

    def boundary_flux(self):
        """
        Evaluate contribution of ds terms to the error estimator as element-wise indicator functions.

        NOTE: The mass matrix is diagonal in P0 space so applying a Jacobi PC is an exact solve!
        """
        if self.bnd_flux_terms == 0:
            print_output("NOTE: No boundary flux terms detected.")
            self.bnd.assign(0.0)
        else:
            mass_term = self.p0test*self.p0trial*dx
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
            solve(mass_term == self.bnd_flux_terms, self.bnd, solver_parameters=params)
        return self.bnd

    def weighted_residual(self):
        """
        Sum the element residual, inter-element flux and boundary flux terms to give the total
        weighted residual.

        If evaluated at the adjoint solution (and time-lagged adjoint solution), yields the so-called
        'Dual Weighted Residual'.
        """
        wr = Function(self.P0, name="Weighted residual")
        wr += self.element_residual()
        wr += self.inter_element_flux()
        wr += self.boundary_flux()
        return wr

    def setup_strong_residual(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in derived class.")

    @property
    def strong_residual(self):
        """
        Evaluate the strong residual.
        """
        import numpy as np
        if not hasattr(self, '_strong_residual_terms'):
            raise ValueError("Cannot evaluate strong residual. Need to set it up first.")
        return np.array([assemble(sr) for sr in list(self._strong_residual_terms)])

    def residual(self):
        raise AttributeError("This method is inherited but unused.")

    def jacobian(self):
        raise AttributeError("This method is inherited but unused.")
