from firedrake import *
from tracer.options import TracerOptions
from adapt.metric import *


class TracerProblem():
    def __init__(self,
                 op=TracerOptions(),
                 stab=None,
                 mesh=None,
                 approach='fixed_mesh',
                 n=2,
                 fe=FiniteElement("Lagrange", triangle, 1),
                 high_order=False):
        assert(fe.family() == 'Lagrange')  # TODO: DG option if finite_element.family() == 'DG'
        if mesh is None:
            self.mesh = SquareMesh(20*n, 20*n, 4, 4)
        else:
            self.mesh = mesh
        self.approach = approach

        # function spaces
        self.V = FunctionSpace(self.mesh, fe)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.n = FacetNormal(self.mesh)
        self.h = CellSize(self.mesh)
        
        # parameters
        self.op = op
        self.op.region_of_interest = [(3., 2., 0.1)]
        self.x0 = 1.
        self.y0 = 2.
        self.r0 = 0.1
        self.nu = Constant(1.)
        self.u = Constant([15., 0.])
        self.params = {'pc_type': 'lu',
                       'mat_type': 'aij' ,
                       'ksp_monitor': None,
                       'ksp_converged_reason': None}
        self.stab = stab if stab is not None else 'no'
        self.high_order = high_order
        
        # outputting
        self.di = 'plots/'
        self.ext = ''
        if self.stab == 'SU':
            self.ext = '_su'
        elif self.stab == 'SUPG':
            self.ext = '_supg'
        self.sol_file = File(self.di + 'sol' + self.ext + '.pvd')
        self.sol_adjoint_file = File(self.di + 'sol_adjoint' + self.ext + '.pvd')
        
    def set_target_vertices(self, rescaling=0.85, num_vertices=None):
        if num_vertices is None:
            num_vertices = self.mesh.num_vertices()
        self.op.target_vertices = num_vertices * rescaling
        
    def source_term(self):
        x, y = SpatialCoordinate(self.mesh)
        cond = And(And(gt(x, self.x0-self.r0), lt(x, self.x0+self.r0)),
                   And(gt(y, self.y0-self.r0), lt(y, self.y0+self.r0)))
        return interpolate(conditional(cond, 1., 0.), self.P0)
    
    def setup_equation(self):
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source_term()

        # finite element problem
        phi = TrialFunction(self.V)
        psi = TestFunction(self.V)
        a = psi*dot(u, grad(phi))*dx
        a += nu*inner(grad(phi), grad(psi))*dx
        a += - nu*psi*dot(n, nabla_grad(phi))*ds(1)
        L = f*psi*dx

        # stabilisation
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
        phi = Function(self.V, name='Tracer concentration')
        solve(self.lhs == self.rhs, phi, bcs=self.bc, solver_parameters=self.params)
        self.sol = phi
        self.sol_file.write(self.sol)
        
    def setup_adjoint_equation(self):
        u = self.u
        nu = self.nu
        n = self.n
        
        # Adjoint source term
        dJdphi = Function(self.P0)
        dJdphi.interpolate(self.op.box(self.mesh))
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
        self.bc_adjoint = DirichletBC(self.V, 0, 1)

    def solve_adjoint(self):
        lam = Function(self.V, name='Adjoint tracer concentration')
        solve(self.lhs_adjoint == self.rhs_adjoint, lam, bcs=self.bc_adjoint, solver_parameters=self.params)
        self.sol_adjoint = lam
        self.sol_adjoint_file.write(self.sol_adjoint)

    def objective_functional(self):
        ks = interpolate(self.op.box(self.mesh), self.P0)
        return assemble(self.sol * ks * dx)

    def get_hessian_metric(self, adjoint=False):
        self.M = steady_metric(self.sol_adjoint if adjoint else self.sol, op=self.op)

    def explicit_estimation(self):
        phi = self.sol
        i = TestFunction(self.P0)
        
        # compute residuals
        self.cell_res = dot(self.u, grad(phi)) - div(self.nu*grad(phi))
        self.edge_res = phi*dot(self.u, self.n) - self.nu*dot(self.n, nabla_grad(phi))
        R = self.cell_res
        r = self.edge_res

        # assemble cell residual
        R_norm = assemble(i*R*R*dx)

        # solve auxiliary problem to assemble edge residual
        r_norm = TrialFunction(self.P0)
        mass_term = i*r_norm*dx
        flux_terms = ((i*r*r)('+') + (i*r*r)('-'))*dS + i*r*r*ds
        r_norm = Function(self.P0)
        solve(mass_term == flux_terms, r_norm)

        # form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit')
 
    def get_isotropic_metric(self):
        name = self.indicator.name()
        eh = project(self.indicator, self.P1)
        eh = normalise_indicator(eh, op=self.op)
        eh.rename(name + '_indicator')
        File(self.di + 'power_' + self.ext + '_indicator_' + name + '.pvd').write(eh)
        self.M = isotropic_metric(eh, op=self.op)
        
    def get_hybrid_metric(self):
        eh = interpolate(Constant(self.mesh.num_vertices()/0.001)*abs(self.indicator), self.P1)
        self.M = scaled_hessian(eh, self.sol, op=self.op)

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
            
            
        # cell residual
        R = (f - dot(u, grad(phi)) + div(nu*grad(phi)))*lam

        # edge residual
        r = TrialFunction(self.P0)
        flux = nu*lam*dot(n, nabla_grad(phi))
        flux_terms = ((i*flux)('+') + (i*flux)('-')) * dS + i*flux*ds(3) + i*flux*ds(4)
        flux_terms += -i*phi*ds(1)
        mass_term = i*r*dx
        r = Function(self.P0)
        solve(mass_term == flux_terms, r)

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
            
        # adjoint source term
        dJdphi = interpolate(self.op.box(self.mesh), self.P0)
            
        # cell residual
        R = (dJdphi + div(u*lam) + div(nu*grad(lam)))*phi
        
        # edge residual
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

    def get_anisotropic_metric(self, adjoint=False):
        try:
            assert self.op.restrict == 'anisotropy'
        except:
            self.op.restrict = 'anisotropy'
            raise Warning("Setting metric restriction method to 'anisotropy'")

        # gradient of adjoint solution
        adj = self.sol_adjoint
        adj_diff = construct_gradient(adj)

        # get fields to take Hessian wrt
        if adjoint:
            F1 = -self.sol*self.u[0] - self.nu*self.sol.dx(0)
            F2 = -self.sol*self.u[1] - self.nu*self.sol.dx(1)
            #source = interpolate(self.op.box(self.mesh), self.P0)
        else:
            F1 = self.sol*self.u[0] - self.nu*self.sol.dx(0)
            F2 = self.sol*self.u[1] - self.nu*self.sol.dx(1)
            #source = self.source_term()

        # construct Hessians
        H1 = construct_hessian(F1, mesh=self.mesh, op=self.op)
        H2 = construct_hessian(F2, mesh=self.mesh, op=self.op)
        #Hf = construct_hessian(source, mesh=self.mesh, op=self.op)

        # form metric
        #self.M = Hf.copy()
        self.M = Function(H1.function_space())
        for i in range(len(adj.dat.data)):
        #    self.M.dat.data[i][:,:] *= adj.dat.data[i]
            self.M.dat.data[i][:,:] += H1.dat.data[i]*adj_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2.dat.data[i]*adj_diff.dat.data[i][1]
        self.M = steady_metric(None, H=self.M, op=self.op)

        # TODO: boundary contributions

        
    def adapt_mesh(self, relaxation_parameter=0.9, prev_metric=None):
        
        # estimate error and generate associated metric
        if self.approach == 'hessian':
            self.get_hessian_metric(adjoint=False)
        elif self.approach == 'hessian_adjoint':
            self.get_hessian_metric(adjoint=True)
        elif self.approach == 'hessian_superposed':
            self.get_hessian_metric(adjoint=False)
            M = self.M.copy()
            self.get_hessian_metric(adjoint=True)
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'explicit':
            self.explicit_estimation()
            self.get_isotropic_metric()
        elif self.approach == 'dwp':
            self.dwp_indication()
            self.get_isotropic_metric()
        elif self.approach == 'dwr':
            self.dwr_estimation()
            self.get_isotropic_metric()
        elif self.approach == 'dwr_adjoint':
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
        elif self.approach == 'dwr_both':
            self.dwr_estimation()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_estimation_adjoint()
            self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
            self.get_isotropic_metric()
        elif self.approach == 'dwr_averaged':
            self.dwr_estimation()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_estimation_adjoint()
            self.indicator.interpolate(Constant(0.5)*(abs(i)+abs(self.indicator)))
            self.get_isotropic_metric()
        elif self.approach == 'dwr_relaxed':
            self.dwr_estimation()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'dwr_superposed':
            self.dwr_estimation()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'dwr_anisotropic':
            self.get_anisotropic_metric(adjoint=False)
        elif self.approach == 'dwr_anisotropic_adjoint':
            self.get_anisotropic_metric(adjoint=True)
        elif self.approach == 'dwr_anisotropic_superposed':
            self.get_anisotropic_metric(adjoint=False)
            M = self.M.copy()
            self.get_anisotropic_metric(adjoint=True)
            self.M = metric_intersection(M, self.M)
        else:
            raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.approach))

        # apply metric relaxation, if requested
        self.M_unrelaxed = self.M.copy()
        if prev_metric is not None:
            self.M.project(metric_relaxation(interp(self.mesh, prev_metric), self.M, relaxation_parameter))
        # (default relaxation of 0.9 following [Power et al 2006])
            
        # adapt mesh
        self.mesh = AnisotropicAdaptation(self.mesh, self.M).adapted_mesh
        #File(self.di + 'mesh_' + mode + '.pvd').write(self.mesh.coordinates)
