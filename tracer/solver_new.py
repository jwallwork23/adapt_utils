from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock       # For extracting adjoint solutions
from fenics_adjoint.projection import ProjectBlock  # Exclude projections from tape reading

import datetime
from time import clock
import numpy as np

from adapt_utils.tracer.options import TracerOptions
from adapt_utils.adapt.metric import *
from adapt_utils.solver import BaseProblem


__all__ = ["SteadyTracerProblem", "OuterLoop"]


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

class SteadyTracerProblem(BaseProblem):
    def __init__(self,
                 op=TracerOptions(),
                 stab=None,
                 mesh=SquareMesh(40, 40, 4, 4),
                 approach='fixed_mesh',
                 discrete_adjoint=False,
                 finite_element=FiniteElement("Lagrange", triangle, 1),
                 high_order=False):
        super(SteadyTracerProblem, self).__init__(mesh, finite_element, approach, stab, True, discrete_adjoint, op, high_order)
        assert(finite_element.family() == 'Lagrange')  # TODO: DG option if finite_element.family() == 'DG'

        # parameters
        self.op.region_of_interest = [(3., 2., 0.1)]
        self.x0 = 1.
        self.y0 = 2.
        self.r0 = 0.1
        self.nu = Constant(1.)
        self.u = Function(self.P1_vec).interpolate(as_vector((15., 0.)))
        self.params = {'pc_type': 'lu',
                       'mat_type': 'aij' ,
                       'ksp_monitor': None,
                       'ksp_converged_reason': None}
        self.gradient_field = self.nu

        # solution fields
        self.solution.rename('Tracer concentration')
        self.adjoint_solution.rename('Adjoint tracer concentration')

        # outputting
        self.di = 'plots/'
        self.ext = ''
        if self.stab == 'SU':
            self.ext = '_su'
        elif self.stab == 'SUPG':
            self.ext = '_supg'
        self.sol_file = File(self.di + 'sol' + self.ext + '.pvd')
        self.sol_adjoint_file = File(self.di + 'sol_adjoint' + self.ext + '.pvd')
        
    def source_term(self):  # TODO: What about other source terms?
        x, y = SpatialCoordinate(self.mesh)
        cond = And(And(gt(x, self.x0-self.r0), lt(x, self.x0+self.r0)),
                   And(gt(y, self.y0-self.r0), lt(y, self.y0+self.r0)))
        return Function(self.P0).interpolate(conditional(cond, 1., 0.))
    
    def solve(self):
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

        # solve
        bc = DirichletBC(self.V, 0, 1)
        solve(a == L, self.solution, bcs=bc, solver_parameters=self.params)
        self.sol_file.write(self.solution)
        
    def solve_continuous_adjoint(self):
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
        
        bc = DirichletBC(self.V, 0, 1)
        solve(a == L, self.adjoint_solution, bcs=bc, solver_parameters=self.params)
        self.sol_adjoint_file.write(self.adjoint_solution)

    def objective_functional(self):
        ks = Function(self.P0).interpolate(self.op.box(self.mesh))
        return assemble(self.solution*ks*dx)

    def get_hessian_metric(self, adjoint=False):
        self.M = steady_metric(self.adjoint_solution if adjoint else self.solution, op=self.op)

    def explicit_estimation(self):
        phi = self.solution
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

    def explicit_estimation_adjoint(self):
        phi = self.solution
        lam = self.adjoint_solution
        u = self.u
        nu = self.nu
        n = self.n
        i = TestFunction(self.P0)

        # cell residual
        R = -div(u*lam) - div(nu*grad(lam))
        R_norm = assemble(i*R*R*dx)

        # edge residual
        r = TrialFunction(self.P0)
        flux = - lam*phi*dot(u, n) - nu*phi*dot(n, nabla_grad(lam))
        flux_terms = ((i*flux*flux)('+') + (i*flux*flux)('-')) * dS
        flux_terms += i*flux*flux*ds(2) + i*flux*flux*ds(3) + i*flux*flux*ds(4)  # Robin BC
        flux_terms += i*lam*lam*ds(1) + i*lam*lam*ds(2)                          # Dirichlet BC
        mass_term = i*r*dx
        r_norm = Function(self.P0)
        solve(mass_term == flux_terms, r_norm)

        # form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit_adjoint')
 
    def solve_high_order(self, adjoint=True):  # TODO: reimplement in base class

        # consider an iso-P2 refined mesh
        fine_mesh = iso_P2(self.mesh)

        # solve adjoint problem on fine mesh using linear elements
        tp_p1 = SteadyTracerProblem(stab=self.stab,
                              mesh=fine_mesh,
                              fe=FiniteElement('Lagrange', triangle, 1))
        if adjoint:
            tp_p1.setup_adjoint_equation()
            tp_p1.solve_adjoint()
        else:
            tp_p1.setup_equation()
            tp_p1.solve()

        # solve adjoint problem on fine mesh using quadratic elements
        tp_p2 = SteadyTracerProblem(stab=self.stab,
                                    mesh=fine_mesh,
                                    fe=FiniteElement('Lagrange', triangle, 2))
        if adjoint:
            tp_p2.setup_adjoint_equation()
            tp_p2.solve_adjoint()
        else:
            tp_p2.setup_equation()
            tp_p2.solve()

        # evaluate difference on fine mesh and project onto coarse mesh
        sol_p1 = tp_p1.sol_adjoint if adjoint else tp_p1.sol
        sol_p2 = tp_p2.sol_adjoint if adjoint else tp_p2.sol
        sol = Function(tp_p2.V).interpolate(sol_p1)
        sol.interpolate(sol_p2 - sol)
        coarse = Function(self.V)
        coarse.project(sol)
        return coarse

    def dwr_estimation(self):
        i = TestFunction(self.P0)
        phi = self.solution
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source_term()
        
        if self.high_order:
            lam = self.solve_high_order(adjoint=True)
        else:
            lam = self.adjoint_solution
            
            
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
        lam = self.adjoint_solution
        u = self.u
        nu = self.nu
        n = self.n
        
        if self.high_order:
            phi = self.solve_high_order(adjoint=False)
        else:
            phi = self.solution
            
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
        
    def get_anisotropic_metric(self, adjoint=False):
        try:
            assert self.op.restrict == 'anisotropy'
        except:
            self.op.restrict = 'anisotropy'
            raise Warning("Setting metric restriction method to 'anisotropy'")

        # solve adjoint problem
        if self.high_order:
            adj = self.solve_high_order(adjoint=not adjoint)
        else:
            adj = self.solution if adjoint else self.adjoint_solution
        sol = self.adjoint_solution if adjoint else self.solution
        adj_diff = construct_gradient(adj)

        # get potential to take Hessian wrt
        x, y = SpatialCoordinate(self.mesh)
        if adjoint:
            source = interpolate(self.op.box(self.mesh), self.P0)
            # F1 = -sol*self.u[0] - self.nu*sol.dx(0) - source*x
            # F2 = -sol*self.u[1] - self.nu*sol.dx(1) - source*y
            F1 = -sol*self.u[0] - self.nu*sol.dx(0)
            F2 = -sol*self.u[1] - self.nu*sol.dx(1)
        else:
            source = self.source_term()
            # F1 = sol*self.u[0] - self.nu*sol.dx(0) - source*x
            # F2 = sol*self.u[1] - self.nu*sol.dx(1) - source*y
            F1 = sol*self.u[0] - self.nu*sol.dx(0)
            F2 = sol*self.u[1] - self.nu*sol.dx(1)

        # NOTES:
        #  * The derivation for the second potential uses the fact that u is divergence free (in
        #    particular, it is constant).
        #  * It is NOT the case that div(F) = f when the commented-out versions of F1 and F2 are
        #    used. In fact:
        #                    div(F)(x) = 1     if x in A
        #                                infty if x in partial A
        #                                0     else
        #    Using these forms lead to high resolution of the boundaries of the source / receiver
        #    regions and low resolution elsewhere.

        # construct Hessians
        H1 = construct_hessian(F1, mesh=self.mesh, op=self.op)
        H2 = construct_hessian(F2, mesh=self.mesh, op=self.op)
        # Hf = construct_hessian(source, mesh=self.mesh, op=self.op)

        # form metric
        self.M = Function(H1.function_space())
        for i in range(len(adj.dat.data)):
            self.M.dat.data[i][:,:] += H1.dat.data[i]*adj_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2.dat.data[i]*adj_diff.dat.data[i][1]
        #     self.M.dat.data[i][:,:] += Hf.dat.dat[i]*adj.dat.data[i]
        self.M = steady_metric(None, H=self.M, op=self.op)

        # TODO: boundary contributions

        
class OuterLoop():
    def __init__(self, approach='hessian', rescaling=0.85, iterates=4, high_order=False, maxit=35, element_rtol=0.005, objective_rtol=0.005):
        self.approach = approach
        self.rescaling = rescaling
        self.iterates = iterates
        self.high_order = high_order
        self.maxit = maxit
        self.element_rtol = element_rtol
        self.objective_rtol = objective_rtol

        self.opt = {}

    def test_meshes(self):
        logfile = open('plots/' + self.approach + '/resolution.log', 'a+')
        logfile.write('\n' + date + '\n\n')
        for i in range(self.iterates):
            print("\nOuter loop {:d}/{:d} for approach '{:s}'".format(i+1, self.iterates, self.approach))
            self.opt[i] = MeshOptimisation(n=2**i,
                                           approach=self.approach,
                                           rescaling=self.rescaling,
                                           high_order=self.high_order,
                                           log=False)
            self.opt[i].maxit = self.maxit
            self.opt[i].element_rtol = self.element_rtol
            self.opt[i].objective_rtol = self.objective_rtol
            self.opt[i].optimise()
            if self.opt[i].maxit_flag:
                self.opt[i].dat['objective'][-1] = np.nan
            logfile.write("loop {:d} elements {:7d} objective {:.4e}\n".format(i, self.opt[i].dat['elements'][-1], self.opt[i].dat['objective'][-1]))
        self.gather()
        logfile.close()

    def test_to_convergence(self):
        logfile = open('plots/' + self.approach + '/full.log', 'a+')
        logfile.write('\n' + date + '\n\n')
        for i in range(self.maxit):
            print("\nOuter loop {:d} for approach '{:s}'".format(i+1, self.approach))
            self.opt[i] = MeshOptimisation(n=i+1,
                                           approach=self.approach,
                                           rescaling=self.rescaling,
                                           high_order=self.high_order,
                                           log=False)
            self.opt[i].maxit = self.maxit
            self.opt[i].element_rtol = self.element_rtol
            self.opt[i].objective_rtol = self.objective_rtol
            self.opt[i].optimise()
            if self.opt[i].maxit_flag:
                self.opt[i].dat['objective'][-1] = np.nan
            logfile.write("loop {:d} elements {:7d} objective {:.4e}\n".format(i, self.opt[i].dat['elements'][-1], self.opt[i].dat['objective'][-1]))

            # convergence criterion
            obj_diff = abs(self.opt[i].dat['objective'][-1] - self.opt[i-1].dat['objective'][-1])
            if obj_diff < self.objective_rtol*self.opt[i-1].dat['objective'][-1]:
                print(self.opt[i].conv_msg.format(i+1, 'convergence in objective functional.'))
                break
        self.gather()
        logfile.close()

    # TODO: Not sure the test_rescaling approach is particularly useful

    def test_rescaling(self):
        logfile = open('plots/' + self.approach + '/rescaling.log', 'a+')
        logfile.write('\n' + date + '\n\n')
        for r, i in zip(np.linspace(0.05, 1., self.iterates), range(self.iterates)):
            self.rescaling = r
            print("\nOuter loop {:d}/{:d} for approach '{:s}'".format(i+1, self.iterates, self.approach))
            self.opt[i] = MeshOptimisation(approach=self.approach,
                                           rescaling=self.rescaling,
                                           high_order=self.high_order,
                                           log=False)
            self.opt[i].maxit = self.maxit
            self.opt[i].element_rtol = self.element_rtol
            self.opt[i].objective_rtol = self.objective_rtol
            self.opt[i].optimise()
            if self.opt[i].maxit_flag:
                self.opt[i].dat['objective'][-1] = np.nan
            logfile.write("loop {:d} elements {:7d} objective {:.4e}\n".format(i, self.opt[i].dat['elements'][-1], self.opt[i].dat['objective'][-1]))
        self.gather()
        logfile.close()
                  
    def scale_to_convergence(self):
        logfile = open('plots/' + self.approach + '/scale_to_convergence.log', 'a+')
        logfile.write('\n' + date + '\n\n')
        logfile.write('maxit: {:d}\n'.format(self.maxit))
        logfile.write('element_rtol: {:.3f}\n'.format(self.element_rtol))
        logfile.write('objective_rtol: {:.3f}\n\n'.format(self.objective_rtol))
        for i in range(self.maxit):
            self.rescaling = float(i+1)*0.4
            print("\nOuter loop {:d} for approach '{:s}'".format(i+1, self.approach))
            self.opt[i] = MeshOptimisation(n=3,
                                           approach=self.approach,
                                           rescaling=self.rescaling,
                                           high_order=self.high_order,
                                           log=False)
            self.opt[i].maxit = self.maxit
            self.opt[i].element_rtol = self.element_rtol
            self.opt[i].objective_rtol = self.objective_rtol
            self.opt[i].optimise()
            if self.opt[i].maxit_flag:
                self.opt[i].dat['objective'][-1] = np.nan
            logfile.write("rescaling {:.2f} elements {:7d} objective {:.4e}\n".format(self.rescaling, self.opt[i].dat['elements'][-1], self.opt[i].dat['objective'][-1]))

            # convergence criterion
            if i > 0:
                obj_diff = abs(self.opt[i].dat['objective'][-1] - self.opt[i-1].dat['objective'][-1])
                if obj_diff < self.objective_rtol*self.opt[i-1].dat['objective'][-1]:
                    print(self.opt[i].conv_msg.format(i+1, 'convergence in objective functional.'))
                    break
        self.gather()
        logfile.close()
    def gather(self):
        N = len(self.opt.keys())
        self.elements = [self.opt[i].dat['elements'][-1] for i in range(N)]
        self.objective = [self.opt[i].dat['objective'][-1] for i in range(N)]
        self.time = [self.opt[i].dat['time'] for i in range(N)]
        self.rescaling = [self.opt[i].rescaling for i in range(N)]
