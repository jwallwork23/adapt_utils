from firedrake import *

from adapt_utils.adapt.metric import *
from adapt_utils.tracer.options import TracerOptions


__all__ = ["TracerProblem"]


class TracerProblem():
    def __init__(self,
                 op=TracerOptions(),
                 stab='no',
                 mesh=None,
                 n=2,
                 fe=FiniteElement("Lagrange", triangle, 1),
                 high_order=False):
        
        # Mesh and function spaces
        assert(fe.family() == 'Lagrange')  # TODO: DG option if finite_element.family() == 'DG'
        if mesh is None:
            self.mesh = RectangleMesh(50*n, 10*n, 50, 10)
        else:
            self.mesh = mesh
        self.V = FunctionSpace(self.mesh, fe)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.n = FacetNormal(self.mesh)
        self.h = CellSize(self.mesh)
        
        # Parameters
        self.op = op
        self.op.restrict = 'num_cells'
        self.op.region_of_interest = [(20., 7.5, 0.5)]  # TODO: Check works without this
        self.x0 = 1.
        self.y0 = 5.
        self.r0 = 0.457
        self.nu = Constant(0.1)
        self.u = Constant([1., 0.])
        self.params = {'pc_type': 'lu', 'mat_type': 'aij' , 'ksp_monitor': None, 'ksp_converged_reason': None}
        self.stab = stab
        self.high_order = high_order
        
        # Outputting
        self.di = 'plots/'
        self.ext = ''
        if self.stab == 'SU':
            self.ext = '_su'
        elif self.stab == 'SUPG':
            self.ext = '_supg'
        self.sol_file = File(self.di + 'stationary_tracer' + self.ext + '.pvd')
        
        # Attributes to be populated
        self.lhs = None
        self.rhs = None
        self.bc = None
        self.lhs_adjoint = None
        self.rhs_adjoint = None
        self.bc_adjoint = None
        
    def set_target_vertices(self, rescaling=0.85, num_vertices=None):
        if num_vertices is None:
            num_vertices = self.mesh.num_vertices()
        self.op.target_vertices = num_vertices * rescaling
        
    def source_term(self):
        x, y = SpatialCoordinate(self.mesh)
        bell = 1 + cos(pi * min_value(sqrt(pow(x - self.x0, 2) + pow(y - self.y0, 2)) / self.r0, 1.0))
        return interpolate(0. + conditional(ge(bell, 0.), bell, 0.), self.P1)

    def setup_equation(self):
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source_term()

        # Finite element problem
        phi = TrialFunction(self.V)
        psi = TestFunction(self.V)
        a = psi*dot(u, grad(phi))*dx
        a += nu*inner(grad(phi), grad(psi))*dx
        a += - nu*psi*dot(n, nabla_grad(phi))*ds(1)
        a += - nu*psi*dot(n, nabla_grad(phi))*ds(2)
        L = f*psi*dx

        # Stabilisation
        if self.stab in ("SU", "SUPG"):
            tau = self.h / (2*sqrt(inner(u, u)))
            stab_coeff = tau * dot(u, grad(psi))
            R_a = dot(u, grad(phi))         # LHS component of strong residual
            if self.stab == 'SUPG':
                R_a += - div(nu*grad(phi))
                R_L = f                     # RHS component of strong residual
                L += stab_coeff*R_L*dx
            a += stab_coeff*R_a*dx
        
        self.lhs = a
        self.rhs = L
        self.bc = DirichletBC(self.V, 0, 1)

    def solve(self):
        if self.lhs is None or self.rhs is None or self.bc is None:
            self.setup_equation()
        phi = Function(self.V, name='Tracer concentration')
        solve(self.lhs == self.rhs, phi, bcs=self.bc, solver_parameters=self.params)
        self.sol = phi
        
    def plot(self):
        self.sol_file.write(self.sol)

    def setup_adjoint_equation(self):
        u = self.u
        nu = self.nu
        n = self.n
        
        # Adjoint source term
        dJdphi = Function(self.P0)
        dJdphi.interpolate(self.op.indicator(self.mesh))
        lam = TrialFunction(self.V)
        psi = TestFunction(self.V)
        
        # Adjoint finite element problem
        a = lam*dot(u, grad(psi))*dx
        a += nu*inner(grad(lam), grad(psi))*dx
        a += -lam*psi*(dot(u, n))*ds(1)
        a += -nu*psi*dot(n, nabla_grad(lam))*ds(1)
        L = dJdphi*psi*dx
        
        # Stabilisation
        if self.stab in ("SU", "SUPG"):
            tau = self.h / (2*sqrt(inner(u, u)))
            stab_coeff = tau * -div(u*psi)
            R_a = -div(u*lam)             # LHS component of strong residual
            if self.stab == "SUPG":
                R_a += -div(nu*grad(lam))
                R_L = dJdphi              # RHS component of strong residual
                L += stab_coeff*R_L*dx
            a += stab_coeff*R_a*dx
        
        self.lhs_adjoint = a
        self.rhs_adjoint = L
        self.bc_adjoint = DirichletBC(self.V, 0, [1, 2])

    def solve_adjoint(self):
        if self.lhs_adjoint is None or self.rhs_adjoint is None or self.bc_adjoint is None:
            self.setup_adjoint_equation()
        lam = Function(self.V, name='Adjoint tracer concentration')
        solve(self.lhs_adjoint == self.rhs_adjoint, lam, bcs=self.bc_adjoint, solver_parameters=self.params)
        self.sol_adjoint = lam

    def plot_adjoint(self):
        File(self.di + 'stationary_tracer' + self.ext + '_adjoint.pvd').write(self.sol_adjoint)

    def objective_functional(self):
        ks = interpolate(self.op.indicator(self.mesh), self.P0)
        return assemble(self.sol * ks * dx)
        
    def get_hessian_metric(self):
        self.M = steady_metric(self.sol, op=self.op)

    def explicit_estimation(self):
        phi = self.sol
        i = TestFunction(self.P0)
        
        # Compute residuals
        self.cell_res = self.source_term() - dot(self.u, grad(phi)) + div(self.nu*grad(phi))
        self.edge_res = -self.nu*dot(self.n, nabla_grad(phi))
        R = self.cell_res
        r = self.edge_res

        # Assemble cell residual
        R_norm = assemble(i*R*R*dx)

        # Solve auxiliary problem to assemble edge residual
        r_norm = TrialFunction(self.P0)
        mass_term = i*r_norm*dx
        flux_terms = ((i*r*r)('+') + (i*r*r)('-'))*dS + i*r*r*ds(3) + i*r*r*ds(4)
        flux_terms += -i*phi*phi*ds(1)
        r_norm = Function(self.P0)
        solve(mass_term == flux_terms, r_norm)

        # Form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit')
        
    def difference_quotient_estimation(self):
        raise NotImplementedError  # TODO
 
    def get_isotropic_metric(self):
        name = self.indicator.name()
        self.indicator = project(self.indicator, self.P1)
        self.indicator = normalise_indicator(self.indicator, op=self.op)
        self.indicator.rename(name + '_indicator')
        self.M = isotropic_metric(self.indicator, op=self.op)

    def plot_indicator(self):
        f = File(self.di + 'stationary_tracer' + self.ext + '_indicator_' + self.indicator.name() + '.pvd')
        f.write(self.indicator)
        
    def dwr_estimation(self):
        i = TestFunction(self.P0)
        phi = self.sol
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source_term()
        
        if self.high_order:
            tp_ho = TracerProblem(stab=self.stab,
                                  mesh=self.mesh,
                                  fe=FiniteElement('Lagrange', triangle, 2))
            tp_ho.setup_adjoint_equation()
            tp_ho.solve_adjoint()
            lam = tp_ho.sol_adjoint - self.sol_adjoint
        else:
            lam = self.sol_adjoint
            
            
        # Cell residual
        self.cell_res = (f - dot(u, grad(phi)) + div(nu*grad(phi)))
        R = self.cell_res*lam

        # Edge residual
        r = TrialFunction(self.P0)
        flux = nu*lam*dot(n, nabla_grad(phi))
        flux_terms = ((i*flux)('+') + (i*flux)('-')) * dS + i*flux*ds(3) + i*flux*ds(4)
        flux_terms += -i*phi*ds(1)
        mass_term = i*r*dx
        r = Function(self.P0)
        solve(mass_term == flux_terms, r)


#         R = f*lam
#         R -= lam*dot(u, grad(phi))
#         R -= nu*inner(grad(phi), grad(lam))
#         flux_terms = 0
#         flux_terms -= - i*nu*lam*dot(n, nabla_grad(phi))*ds(1)
#         flux_terms -= - i*nu*lam*dot(n, nabla_grad(phi))*ds(2)

#         # Stabilisation
#         if self.stab in ("SU", "SUPG"):
#             tau = self.h / (2*sqrt(inner(u, u)))
#             stab_coeff = tau * dot(u, grad(lam))
#             R_a = dot(u, grad(phi))         # LHS component of strong residual
#             if self.stab == 'SUPG':
#                 R_a += - div(nu*grad(phi))
#                 R_L = f                     # RHS component of strong residual
#                 R += stab_coeff*R_L
#             R -= stab_coeff*R_a
            
#         r = TrialFunction(self.P0)
#         mass_term = i*r*dx
#         r = Function(self.P0)
#         solve(mass_term == flux_terms, r)
        

        self.cell_res = R
        self.edge_res = r
        self.indicator = project(R + r, self.P0)
        self.indicator.rename('dwr')
        
    def dwr_estimation_adjoint(self):
        i = TestFunction(self.P0)
        lam = self.sol_adjoint
        u = self.u
        nu = self.nu
        n = self.n
        
        if self.high_order:
            tp_ho = TracerProblem(stab=self.stab,
                                  mesh=self.mesh,
                                  fe=FiniteElement('Lagrange', triangle, 2))
            tp_ho.setup_equation()
            tp_ho.solve()
            phi = tp_ho.sol - self.sol
        else:
            phi = self.sol
            
        # Adjoint source term
        dJdphi = Function(self.P0)
        dJdphi.interpolate(self.op.indicator(self.mesh))
            
        # Cell residual
        self.cell_res_adjoint = (dJdphi + div(u*lam) + div(nu*grad(lam)))
        R = self.cell_res_adjoint * phi
        
        # Edge residual
        r = TrialFunction(self.P0)
        flux = - lam*phi*dot(u, n) - nu*phi*dot(n, nabla_grad(lam))
        flux_terms = ((i*flux)('+') + (i*flux)('-')) * dS
        flux_terms += i*flux*ds(2) + i*flux*ds(3) + i*flux*ds(4)  # Robin BC
        flux_terms += -i*lam*ds(1) -i*lam*ds(2)                   # Dirichlet BC
        mass_term = i*r*dx
        r = Function(self.P0)
        solve(mass_term == flux_terms, r)

        self.cell_res_adjoint = R
        self.edge_res_adjoint = r
        self.indicator = project(R + r, self.P0)
        self.indicator.rename('dwr_adjoint')
        
    def dwp_indication(self):
        self.indicator = project(self.sol * self.sol_adjoint, self.P0)
        self.indicator.rename('dwp')
        
    def adapt_mesh(self, mode='hessian', relaxation_parameter=0.9, prev_metric=None, plot_mesh=True):
        
        # Estimate error and generate associated metric
        if mode == 'hessian':
            self.get_hessian_metric()
        elif mode == 'explicit':
            self.explicit_estimation()
            self.get_isotropic_metric()
        elif mode == 'dwp':
            self.dwp_indication()
            self.get_isotropic_metric()
        elif mode == 'dwr':
            self.dwr_estimation()
            self.get_isotropic_metric()
        elif mode == 'dwr_adjoint':
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
        elif mode == 'dwr_both':
            self.dwr_estimation()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_estimation_adjoint()
            self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
            self.get_isotropic_metric()
        elif mode == 'dwr_averaged':
            self.dwr_estimation()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_estimation_adjoint()
            self.indicator.interpolate(Constant(0.5)*(abs(i)+abs(self.indicator)))
            self.get_isotropic_metric()
        elif mode == 'dwr_relaxed':
            self.dwr_estimation()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
            self.M = metric_relaxation(M, self.M)
        elif mode == 'dwr_superposed':
            self.dwr_estimation()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
            self.M = metric_intersection(M, self.M)
        else:
            raise ValueError("Adaptivity mode {:s} not regcognised.".format(mode))

        # Apply metric relaxation, if requested
        self.M_unrelaxed = self.M.copy()
        if prev_metric is not None:
            self.M.assign(metric_relaxation(project(prev_metric, self.M.function_space()), self.M, relaxation_parameter))
        # (Default relaxation of 0.9 following [Power et al 2006])
            
        # Adapt mesh
        self.mesh = adapt(self.mesh, self.M)
        if plot_mesh:
            f = File(self.di + 'stationary_tracer' + self.ext + '_mesh_' + mode + '.pvd')
            f.write(self.mesh.coordinates)
