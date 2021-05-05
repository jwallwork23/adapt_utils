r"""
Exner equation

2D conservation of mass equation describing bed evolution due to sediment transport

The equation reads

.. math::
    \frac{\partial z_b}{\partial t} + (morfac/(1-p)) \nabla_h \cdot (Q_b)
    = (morfac/(1-p)) H ((Sink S) - Source)
    :label: exner_eq

where :math:'z_b' is the bedlevel, :math:'S' is :math:'q=HT' for conservative
and :math:'T' for non-conservative, :math:`\nabla_h` denotes horizontal gradient,
:math:'morfac' is the morphological scale factor, :math:'p' is the porosity and
:math:'Q_b' is the bedload transport vector

"""

from __future__ import absolute_import
from thetis.equation import Term, Equation
from thetis.utility import *

__all__ = [
    'ExnerEquation',
    'ExnerTerm',
    'ExnerSourceTerm',
    'ExnerBedloadTerm'
]


class ExnerTerm(Term):
    """
    Generic term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space, depth, sed_model, conservative=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg sed_model: :class: `SedimentModel` containing sediment info
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(ExnerTerm, self).__init__(function_space)
        self.n = FacetNormal(self.mesh)
        self.depth = depth
        self.sed_model = sed_model

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree)
        self.dS = dS(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)
        self.conservative = conservative


class ExnerSourceTerm(ExnerTerm):
    r"""
    Source term accounting for suspended sediment transport

    The weak form reads

    .. math::
        F_s = \int_\Omega (\sigma - sediment * \phi) * depth \psi dx

    where :math:`\sigma` is a user defined source scalar field :class:`Function`
    and :math:`\phi` is a user defined source scalar field :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        sediment = fields.get('sediment')
        source = fields.get('source')
        sink = fields.get('sink')
        morfac = fields.get('morfac')
        porosity = fields.get('porosity')

        fac = Constant(morfac/(1.0-porosity))
        H = self.depth.get_total_depth(fields_old['elev_2d'])

        source_dep = source*H

        if self.conservative:
            sink_dep = sink
        else:
            sink_dep = sink*H

        f = inner(fac*(source_dep-sediment*sink_dep), self.test)*self.dx

        return f

class ExnerBedloadTerm(ExnerTerm):
    r"""
    Bedload transport term, \nabla_h \cdot \textbf{Q_b}
    The weak form is
    .. math::
        \int_\Omega  \nabla_h \cdot \textbf{Q_b} \psi  dx
        = - \int_\Omega (\textbf{Q_b} \cdot \nabla) \psi dx
        + \int_\Gamma \psi \textbf{Q_b} \cdot \textbf{n} dS
    where :math:`\textbf{n}` is the unit normal of the element interfaces.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        qbx, qby = self.sed_model.get_bedload_term(solution)

        morfac = fields.get('morfac')
        porosity = fields.get('porosity')

        fac = Constant(morfac/(1.0-porosity))

        # bnd_conditions are the shallow water bcs, any boundary for which
        # nothing is specified is assumed closed

        for bnd_marker in (bnd_conditions or []):
            no_contr = False
            keys = [*bnd_conditions[bnd_marker].keys()]
            values = [*bnd_conditions[bnd_marker].values()]
            for i in range(len(keys)):
                if keys[i] != ('elev' or 'uv'):
                    if float(values[i]) == 0.0:
                        no_contr = True
                elif keys[i] == 'uv':
                    if all(j == 0.0 for j in [float(j) for j in values[i]]):
                        no_contr = True
            if not no_contr:
                f += -self.test*(fac*qbx*self.n[0] + fac*qby*self.n[1])*self.ds(bnd_marker)

        f += (fac*qbx*self.test.dx(0) + fac*qby*self.test.dx(1))*self.dx

        return -f

class ExnerSedimentSlideTerm(ExnerTerm):
    r"""
    TO DO
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        diff_tensor = self.sed_model.get_sediment_slide_term(solution)

        diff_flux = dot(diff_tensor, grad(-solution))
        f += inner(grad(self.test), diff_flux)*dx
        f += -avg(self.sed_model.sigma)*inner(jump(self.test, self.sed_model.n),dot(avg(diff_tensor), jump(solution, self.sed_model.n)))*dS
        f += -inner(avg(dot(diff_tensor, grad(self.test))),jump(solution, self.sed_model.n))*dS
        f += -inner(jump(self.test, self.sed_model.n), avg(dot(diff_tensor, grad(solution))))*dS

        return -f

class ExnerEquation(Equation):
    """
    Exner equation

    2D conservation of mass equation describing bed evolution due to sediment transport
    """
    def __init__(self, function_space, depth, sed_model, conservative):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg sed_model: :class: `SedimentModel` containing sediment info
        :kwarg bool conservative: whether to use conservative tracer
        """
        super().__init__(function_space)
        if sed_model is None:
            raise ValueError('To use the exner equation must define a sediment model')
        args = (function_space, depth, sed_model, conservative)
        if sed_model.solve_suspended_sediment:
            self.add_term(ExnerSourceTerm(*args), 'source')
        if sed_model.use_bedload:
            self.add_term(ExnerBedloadTerm(*args), 'implicit')
        if sed_model.use_sediment_slide:
            self.add_term(ExnerSedimentSlideTerm(*args), 'implicit')
