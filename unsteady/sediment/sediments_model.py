from thetis import *


class Corrective_Velocity_Factor:
    def __init__(self, depth, ksp, ks, settling_velocity, ustar):
        self.ksp = ksp
        self.ks = ks
        self.settling_velocity = settling_velocity

        self.a = Constant(self.ks/2)

        self.kappa = physical_constants['von_karman']

        self.depth = depth
        self.ustar = ustar

        # correction factor to advection velocity in sediment concentration equation
        self.Bconv = conditional(self.depth > Constant(1.1)*self.ksp, self.ksp/self.depth, Constant(1/1.1))
        self.Aconv = conditional(self.depth > Constant(1.1)*self.a, self.a/self.depth, Constant(1/1.1))

        # take max of value calculated either by ksp or depth
        self.Amax = conditional(self.Aconv > self.Bconv, self.Aconv, self.Bconv)

        self.r1conv = Constant(1) - (1/self.kappa)*conditional(self.settling_velocity/self.ustar < Constant(1), self.settling_velocity/self.ustar, Constant(1))

        self.Ione = conditional(self.r1conv > Constant(1e-8), (Constant(1) - self.Amax**self.r1conv)/self.r1conv, conditional(self.r1conv < Constant(- 1e-8), (Constant(1) - self.Amax**self.r1conv)/self.r1conv, ln(self.Amax)))

        self.Itwo = conditional(self.r1conv > Constant(1e-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, conditional(self.r1conv < Constant(- 1e-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, Constant(-0.5)*ln(self.Amax)**2))

        self.alpha = -(self.Itwo - (ln(self.Amax) - ln(30))*self.Ione)/(self.Ione * ((ln(self.Amax) - ln(30)) + Constant(1)))

        # final correction factor
        self.corr_vel_factor = Function(depth.function_space()).interpolate(conditional(conditional(self.alpha > Constant(1), Constant(1), self.alpha) < Constant(0), Constant(0), conditional(self.alpha > Constant(1), Constant(1), self.alpha)))

    def update(self):

        # final correction factor
        self.corr_vel_factor.interpolate(conditional(conditional(self.alpha > Constant(1), Constant(1), self.alpha) < Constant(0), Constant(0), conditional(self.alpha > Constant(1), Constant(1), self.alpha)))


class SedimentModel(object):
    def __init__(self, options, suspendedload, convectivevel,
                 bedload, angle_correction, slope_eff, seccurrent,
                 mesh2d, bathymetry_2d, uv_init, elev_init, ks, average_size, porosity=0.4, max_angle=None, meshgrid_size=None,
                 dt=None, beta_fn=1.3, surbeta2_fn=1/1.5, alpha_secc_fn=0.75, viscosity_morph=1e-6,
                 morfac = 1, wetting_and_drying=False, wetting_alpha=0.1, rhos=2650, cons_tracer=False, sediment_slide=False):

        """
        Set up a full morphological model simulation using as an initial condition the results of a hydrodynamic only model.

        Inputs:
        options - solver_obj options
        suspendedload - switch to turn on suspended sediment transport
        convectivevel - switch on convective velocity correction factor in sediment concentration equation
        bedload - switch to turn on bedload transport
        angle_correction - switch on slope effect angle correction
        slope_eff - switch on slope effect magnitude correction
        seccurrent - switch on secondary current for helical flow effect
        sediment_slide - switch on sediment slide mechanism
        mesh2d - define mesh working on
        bathymetry2d - define bathymetry of problem
        uv_init - initial velocity
        elev_init - initial elevation
        ks - bottom friction coefficient for quadratic drag coefficient
        average_size - average sediment size
        beta_fn - magnitude slope effect parameter
        surbeta2_fn - angle correction slope effect parameter
        alpha_secc_fn - secondary current parameter
        viscosity_morph - viscosity value in morphodynamic equations
        wetting_and_drying - wetting and drying switch
        wetting_alpha - wetting and drying parameter
        rhos - sediment density
        cons_tracer - conservative tracer switch

        """

        self.suspendedload = suspendedload
        self.cons_tracer = cons_tracer
        self.convectivevel = convectivevel
        self.bedload = bedload
        self.angle_correction = angle_correction
        self.slope_eff = slope_eff
        self.seccurrent = seccurrent
        self.use_sediment_slide = sediment_slide
        self.wetting_and_drying = wetting_and_drying

        self.morfac = morfac
        self.dt = dt
        self.average_size = average_size
        self.ks = ks
        self.wetting_alpha = wetting_alpha
        self.rhos = rhos
        self.porosity = porosity
        self.meshgrid_size = meshgrid_size
        self.max_angle = max_angle
        self.uv_init = uv_init
        self.elev_init = elev_init

        self.options = options

        self.bathymetry_2d = bathymetry_2d

        self.t_old = Constant(0.0)

        # define function spaces
        self.P1_2d = get_functionspace(mesh2d, "DG", 1)
        self.V = get_functionspace(mesh2d, "CG", 1)
        self.vector_cg = VectorFunctionSpace(mesh2d, "CG", 1)

        self.n = FacetNormal(mesh2d)

        # define parameters
        self.g = physical_constants['g_grav']
        self.rhow = physical_constants['rho0']
        self.kappa = physical_constants['von_karman']

        self.ksp = Constant(3*self.average_size)
        self.a = Constant(self.ks/2)
        self.viscosity = Constant(viscosity_morph)

        # magnitude slope effect parameter
        self.beta = Constant(beta_fn)
        # angle correction slope effect parameters
        self.surbeta2 = Constant(surbeta2_fn)
        # secondary current parameter
        self.alpha_secc = Constant(alpha_secc_fn)

        # calculate critical shields parameter thetacr
        self.R = Constant(self.rhos/self.rhow - 1)

        self.dstar = Constant(self.average_size*((self.g*self.R)/(self.viscosity**2))**(1/3))
        if max(self.dstar.dat.data[:] < 1):
            print('ERROR: dstar value less than 1')
        elif max(self.dstar.dat.data[:] < 4):
            self.thetacr = Constant(0.24*(self.dstar**(-1)))
        elif max(self.dstar.dat.data[:] < 10):
            self.thetacr = Constant(0.14*(self.dstar**(-0.64)))
        elif max(self.dstar.dat.data[:] < 20):
            self.thetacr = Constant(0.04*(self.dstar**(-0.1)))
        elif max(self.dstar.dat.data[:] < 150):
            self.thetacr = Constant(0.013*(self.dstar**(0.29)))
        else:
            self.thetacr = Constant(0.055)

        # critical bed shear stress
        self.taucr = Constant((self.rhos-self.rhow)*self.g*self.average_size*self.thetacr)

        # calculate settling velocity
        if self.average_size <= 1e-04:
            self.settling_velocity = Constant(self.g*(self.average_size**2)*self.R/(18*self.viscosity))
        elif self.average_size <= 1e-03:
            self.settling_velocity = Constant((10*self.viscosity/self.average_size)*(sqrt(1 + 0.01*((self.R*self.g*(self.average_size**3))/(self.viscosity**2)))-1))
        else:
            self.settling_velocity = Constant(1.1*sqrt(self.g*self.average_size*self.R))

        self.uv_cg = Function(self.vector_cg).project(self.uv_init)

        # define bed gradient
        self.old_bathymetry_2d = Function(self.V).interpolate(self.bathymetry_2d)
        self.dzdx = self.old_bathymetry_2d.dx(0)
        self.dzdy = self.old_bathymetry_2d.dx(1)

        self.depth_expr = DepthExpression(self.bathymetry_2d, use_wetting_and_drying=self.wetting_and_drying, wetting_and_drying_alpha=self.wetting_alpha)
        self.depth = Function(self.V).project(self.depth_expr.get_total_depth(self.elev_init))

        self.horizontal_velocity = self.uv_cg[0]
        self.vertical_velocity = self.uv_cg[1]

        # define bed friction
        self.hc = conditional(self.depth > Constant(0.001), self.depth, Constant(0.001))
        self.aux = conditional(11.036*self.hc/self.ks > Constant(1.001), 11.036*self.hc/self.ks, Constant(1.001))
        self.qfc = Constant(2)/(ln(self.aux)/self.kappa)**2
        # skin friction coefficient
        self.cfactor = conditional(self.depth > self.ksp, Constant(2)*(((1/self.kappa)*ln(11.036*self.depth/self.ksp))**(-2)), Constant(0.0))
        # mu - ratio between skin friction and normal friction
        self.mu = conditional(self.qfc > Constant(0), self.cfactor/self.qfc, Constant(0))

        # calculate bed shear stress
        self.unorm = (self.horizontal_velocity**2) + (self.vertical_velocity**2)
        self.bed_stress = Function(self.V).interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        options.solve_exner = True

        if self.suspendedload:
            # deposition flux - calculating coefficient to account for stronger conc at bed
            self.B = conditional(self.a > self.depth, Constant(1.0), self.a/self.depth)
            self.ustar = sqrt(Constant(0.5)*self.qfc*self.unorm)
            self.rouse_number = (self.settling_velocity/(self.kappa*self.ustar)) - Constant(1)

            self.intermediate_step = conditional(abs(self.rouse_number) > Constant(1e-04),
                                                 self.B*(Constant(1)-self.B**min_value(self.rouse_number, Constant(3)))/min_value(self.rouse_number,
                                                 Constant(3)), -self.B*ln(self.B))

            self.integrated_rouse = max_value(conditional(self.intermediate_step > Constant(1e-12), Constant(1)/self.intermediate_step,
                                                          Constant(1e12)), Constant(1))

            # erosion flux - above critical velocity bed is eroded
            self.s0 = (conditional(self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu > Constant(0), self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu, Constant(0)) - self.taucr)/self.taucr
            self.ceq = Function(self.P1_2d).project(Constant(0.015)*(self.average_size/self.a) * ((conditional(self.s0 < Constant(0), Constant(0), self.s0))**(1.5))/(self.dstar**0.3))

            if self.convectivevel:
                self.corr_factor_model = Corrective_Velocity_Factor(self.depth, self.ksp, self.ks, self.settling_velocity, self.ustar)
            # update sediment rate to ensure equilibrium at inflow
            if self.cons_tracer:
                self.equiltracer = Function(self.P1_2d).interpolate(self.depth*self.ceq/self.integrated_rouse)
            else:
                self.equiltracer = Function(self.P1_2d).interpolate(self.ceq/self.integrated_rouse)

            # get individual terms
            self.depo = self.settling_velocity*self.integrated_rouse
            self.ero = Function(self.P1_2d).project(self.settling_velocity*self.ceq)

            self.depo_term = Function(self.P1_2d).project(self.depo/self.depth)
            self.ero_term = Function(self.P1_2d).project(self.ero/self.depth)

            # calculate depth-averaged source term for sediment concentration equation
            if self.cons_tracer:
                self.source_exp = Function(self.P1_2d).project(-(self.depo*self.equiltracer/(self.depth**2)) + (self.ero/self.depth))
            else:
                self.source_exp = Function(self.P1_2d).project(-(self.depo*self.equiltracer/self.depth) + (self.ero/self.depth))

            self.options.solve_sediment = True
            self.options.use_tracer_conservative_form = self.cons_tracer
            if self.convectivevel:
                self.options.tracer_advective_velocity_factor = self.corr_factor_model.corr_vel_factor
        else:
            self.options.solve_tracer = False
        if self.bedload:
            # calculate angle of flow
            self.calfa = Function(self.V).interpolate(self.horizontal_velocity/sqrt(self.unorm))
            self.salfa = Function(self.V).interpolate(self.vertical_velocity/sqrt(self.unorm))
            if self.angle_correction:
                # slope effect angle correction due to gravity
                self.stress = Function(self.V).interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

    def get_bedload_term(self, solution):

        if self.slope_eff:
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            # we use z_n1 and equals so that we can use an implicit method in Exner
            self.slopecoef = Constant(1) + self.beta*(solution.dx(0)*self.calfa + solution.dx(1)*self.salfa)
        else:
            self.slopecoef = Constant(1.0)

        if self.angle_correction:
            # slope effect angle correction due to gravity
            self.cparam = Function(self.V).interpolate((self.rhos-self.rhow)*self.g*self.average_size*(self.surbeta2**2))
            self.tt1 = conditional(self.stress > Constant(1e-10), sqrt(self.cparam/self.stress), sqrt(self.cparam/Constant(1e-10)))

            # add on a factor of the bed gradient to the normal
            self.aa = self.salfa + self.tt1*self.dzdy
            self.bb = self.calfa + self.tt1*self.dzdx

            self.comb = sqrt(self.aa**2 + self.bb**2)
            self.norm = conditional(self.comb > Constant(1e-10), self.comb, Constant(1e-10))

            # we use z_n1 and equals so that we can use an implicit method in Exner
            self.calfamod = (self.calfa + (self.tt1*solution.dx(0)))/self.norm
            self.salfamod = (self.salfa + (self.tt1*solution.dx(1)))/self.norm

        if self.seccurrent:
            # accounts for helical flow effect in a curver channel
            # use z_n1 and equals so can use an implicit method in Exner
            self.free_surface_dx = self.depth.dx(0) - solution.dx(0)
            self.free_surface_dy = self.depth.dx(1) - solution.dx(1)

            self.velocity_slide = (self.horizontal_velocity*self.free_surface_dy)-(self.vertical_velocity*self.free_surface_dx)

            self.tandelta_factor = Constant(7)*self.g*self.rhow*self.depth*self.qfc/(Constant(2)*self.alpha_secc*((self.horizontal_velocity**2) + (self.vertical_velocity**2)))

            # accounts for helical flow effect in a curver channel
            if self.angle_correction:
                # if angle has already been corrected we must alter the corrected angle to obtain the corrected secondary current angle
                self.t_1 = (self.bed_stress*self.slopecoef*self.calfamod) + (self.vertical_velocity*self.tandelta_factor*self.velocity_slide)
                self.t_2 = (self.bed_stress*self.slopecoef*self.salfamod) - (self.horizontal_velocity*self.tandelta_factor*self.velocity_slide)
            else:
                self.t_1 = (self.bed_stress*self.slopecoef*self.calfa) + (self.vertical_velocity*self.tandelta_factor*self.velocity_slide)
                self.t_2 = ((self.bed_stress*self.slopecoef*self.salfa) - (self.horizontal_velocity*self.tandelta_factor*self.velocity_slide))

            # calculated to normalise the new angles
            self.t4 = sqrt((self.t_1**2) + (self.t_2**2))

            # updated magnitude correction and angle corrections
            self.slopecoef_secc = self.t4/self.bed_stress

            self.calfanew = self.t_1/self.t4
            self.salfanew = self.t_2/self.t4

        # implement meyer-peter-muller bedload transport formula
        self.thetaprime = self.mu*(self.rhow*Constant(0.5)*self.qfc*self.unorm)/((self.rhos-self.rhow)*self.g*self.average_size)

        # if velocity above a certain critical value then transport occurs
        self.phi = conditional(self.thetaprime < self.thetacr, 0, Constant(8)*(self.thetaprime-self.thetacr)**1.5)

        # bedload transport flux with magnitude correction
        if self.seccurrent:
            self.qb_total = self.slopecoef_secc*self.phi*sqrt(self.g*self.R*self.average_size**3)
        else:
            self.qb_total = self.slopecoef*self.phi*sqrt(self.g*self.R*self.average_size**3)

        # formulate bedload transport flux with correct angle depending on corrections implemented
        if self.angle_correction and self.seccurrent is False:
            self.qbx = self.qb_total*self.calfamod
            self.qby = self.qb_total*self.salfamod
        elif self.seccurrent:
            self.qbx = self.qb_total*self.calfanew
            self.qby = self.qb_total*self.salfanew
        else:
            self.qbx = self.qb_total*self.calfa
            self.qby = self.qb_total*self.salfa

        return self.qbx, self.qby

    def get_sediment_slide_term(self, bathymetry):
        # add component to bedload transport to ensure the slope angle does not exceed a certain value
        mesh2d = bathymetry.function_space().mesh()
        # maximum gradient allowed by sediment slide mechanism
        self.tanphi = tan(self.max_angle*pi/180)
        # approximate mesh step size for sediment slide mechanism
        L = self.meshgrid_size

        degree_h = self.P1_2d.ufl_element().degree()

        if degree_h == 0:
            self.sigma = 1.5 / CellSize(mesh2d)
        else:
            self.sigma = 5.0*degree_h*(degree_h + 1)/CellSize(mesh2d)

        # define bed gradient
        dzdx = bathymetry.dx(0)
        dzdy = bathymetry.dx(1)

        # calculate normal to the bed
        nz = 1/sqrt(1 + (dzdx**2 + dzdy**2))

        betaangle = asin(sqrt(1 - (nz**2)))
        self.tanbeta = sqrt(1 - (nz**2))/nz

        # calculating magnitude of added component
        qaval = conditional(self.tanbeta - self.tanphi > 0, (1-self.porosity)*0.5*(L**2)*(self.tanbeta - self.tanphi)/(cos(betaangle*self.dt*self.morfac)), 0)
        # multiplying by direction
        alphaconst = conditional(sqrt(1 - (nz**2)) > 0, - qaval*(nz**2)/sqrt(1 - (nz**2)), 0)

        diff_tensor = as_matrix([[alphaconst, 0, ], [0, alphaconst, ]])

        return diff_tensor

    def update(self, fwd_solution, fwd_solution_bathymetry):
        # update bathymetry
        self.old_bathymetry_2d.interpolate(fwd_solution_bathymetry)
        # extract new elevation and velocity and project onto CG space
        self.uv1, self.elev1 = fwd_solution.split()
        self.uv_cg.project(self.uv1)

        self.bed_stress.interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        self.depth_expr = DepthExpression(self.old_bathymetry_2d, use_wetting_and_drying=self.wetting_and_drying, wetting_and_drying_alpha=self.wetting_alpha)
        self.depth.project(self.depth_expr.get_total_depth(self.elev1))

        if self.suspendedload:
            # erosion flux - above critical velocity bed is eroded
            self.ceq.project(Constant(0.015)*(self.average_size/self.a) * ((conditional(self.s0 < Constant(0), Constant(0), self.s0))**(1.5))/(self.dstar**0.3))

            self.ero.project(self.settling_velocity*self.ceq)
            self.ero_term.project(self.ero/self.depth)
            self.depo_term.project(self.depo/self.depth)

            # update sediment rate to ensure equilibrium at inflow
            if self.cons_tracer:
                self.equiltracer.interpolate(self.depth*self.ceq/self.integrated_rouse)
            else:
                self.equiltracer.interpolate(self.ceq/self.integrated_rouse)

        if self.bedload:
            # calculate angle of flow
            self.calfa.interpolate(self.uv_cg[0]/sqrt(self.unorm))
            self.salfa.interpolate(self.uv_cg[1]/sqrt(self.unorm))

            if self.angle_correction:
                # slope effect angle correction due to gravity
                self.stress.interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)
                self.cparam.interpolate((self.rhos-self.rhow)*self.g*self.average_size*(self.surbeta2**2))
            if self.convectivevel:
                self.corr_factor_model.update()
