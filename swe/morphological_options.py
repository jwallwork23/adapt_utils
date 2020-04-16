from thetis import *
from thetis.configuration import *

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["MorphOptions"]


class MorphOptions(ShallowWaterOptions):
    """
    Parameter class for general morphological problems.
    """
    def set_up_suspended(self, mesh, tracer=None):
        P1 = FunctionSpace(mesh, "CG", 1)
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)
        P1DG = FunctionSpace(mesh, "DG", 1)
        P1DG_vec = VectorFunctionSpace(mesh, "DG", 1)

        R = Constant(2650/1000 - 1)
        self.dstar = Constant(self.average_size*((self.g*R)/(self.base_viscosity**2))**(1/3))
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
            
        self.taucr = Constant((2650-1000)*self.gravity*self.average_size*self.thetacr)
        
        if self.average_size <= 100*(10**(-6)):
            self.settling_velocity = Constant(9.81*(self.average_size**2)*((2650/1000)-1)/(18*self.base_viscosity))
        elif self.average_size <= 1000*(10**(-6)):
            self.settling_velocity = Constant((10*self.base_viscosity/self.average_size)*(sqrt(1 + 0.01*((((2650/1000) - 1)*9.81*(self.average_size**3))/(self.base_viscosity**2)))-1))
        else:
            self.settling_velocity = Constant(1.1*sqrt(9.81*self.average_size*((2650/1000) - 1)))                
        self.uv_d = project(self.uv_d, P1DG_vec)
        self.eta_d = project(self.eta_d, P1DG)
        
        self.u_cg = project(self.uv_d, P1_vec)
        self.horizontal_velocity = project(self.u_cg[0], P1)
        self.vertical_velocity = project(self.u_cg[1], P1)
        self.elev_cg = project(self.eta_d, P1)
        
        if self.t_old.dat.data[:] == 0.0:
            self.set_bathymetry(P1)
        else:
            self.bathymetry = project(self.bathymetry, P1)

        self.depth = project(self.elev_cg + self.bathymetry, P1)
    
        self.unorm = project(self.horizontal_velocity**2 + self.vertical_velocity**2, P1DG)

        self.hc = conditional(self.depth > 0.001, self.depth, 0.001)
        self.aux = conditional(11.036*self.hc/self.ks > 1.001, 11.036*self.hc/self.ks, 1.001)
        self.qfc = 2/(ln(self.aux)/0.4)**2
        
        self.TOB = project(1000*0.5*self.qfc*self.unorm, P1)
        
        
        # skin friction coefficient
        
        self.cfactor = project(self.get_cfactor(), P1DG)
        # mu - ratio between skin friction and normal friction
        self.mu = project(conditional(self.qfc > 0, self.cfactor/self.qfc, 0), P1DG)
        
        
        self.a = (self.ks)/2
        self.B = project(conditional(self.a > self.depth, 1, self.a/self.depth), P1DG)
        self.ustar = project(sqrt(0.5*self.qfc*self.unorm), P1DG)
        self.exp1 = project(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), conditional((self.settling_velocity/(0.4*self.ustar)) -1 > 3, 3, (self.settling_velocity/(0.4*self.ustar))-1), 0), P1DG)
        self.coefftest = project(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), self.B*(1-self.B**self.exp1)/self.exp1, -self.B*ln(self.B)), P1DG)
        self.coeff = project(conditional(self.coefftest>0, 1/self.coefftest, 0), P1DG)
        
        
        # erosion flux - for vanrijn
        self.s0 = project((conditional(1000*0.5*self.qfc*self.unorm*self.mu > 0, 1000*0.5*self.qfc*self.unorm*self.mu, 0) - self.taucr)/self.taucr, P1DG)
        self.ceq = project(0.015*(self.average_size/self.a) * ((conditional(self.s0 < 0, 0, self.s0))**(1.5))/(self.dstar**0.3), P1DG)
        
        self.tracer_init = project(self.ceq/self.coeff, P1DG)
        
        
        self.tracer_init_value = Constant(self.ceq.at([0,0])/self.coeff.at([0,0]))
        self.source = project(self.set_source_tracer(P1DG, solver_obj = None, init = True, t_old = self.t_old, tracer = tracer), P1DG)
        self.qbsourcedepth = project(self.source * self.depth, P1)
        
        if self.convective_vel_flag:
            # correction factor to advection velocity in sediment concentration equation

            self.Bconv = interpolate(conditional(self.depth > 1.1*self.ksp, self.ksp/self.depth, self.ksp/(1.1*self.ksp)), P1DG)
            self.Aconv = interpolate(conditional(self.depth > 1.1* self.a, self.a/self.depth, self.a/(1.1*self.a)), P1DG)
                    
            # take max of value calculated either by ksp or depth
            self.Amax = interpolate(conditional(self.Aconv > self.Bconv, self.Aconv, self.Bconv), P1DG)

            self.r1conv = interpolate(1 - (1/0.4)*conditional(self.settling_velocity/self.ustar < 1, self.settling_velocity/self.ustar, 1), P1DG)

            self.Ione = interpolate(conditional(self.r1conv > 10**(-8), (1 - self.Amax**self.r1conv)/self.r1conv, conditional(self.r1conv < - 10**(-8), (1 - self.Amax**self.r1conv)/self.r1conv, ln(self.Amax))), P1DG)

            self.Itwo = interpolate(conditional(self.r1conv > 10**(-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, conditional(self.r1conv < - 10**(-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, -0.5*ln(self.Amax)**2)), P1DG)

            self.alpha = interpolate(-(self.Itwo - (ln(self.Amax) - ln(30))*self.Ione)/(self.Ione * ((ln(self.Amax) - ln(30)) + 1)), P1DG)

            # final correction factor
            self.corrective_velocity_factor = Function(self.P1DG).interpolate(conditional(conditional(self.alpha > 1, 1, self.alpha) < 0, 0, conditional(self.alpha > 1, 1, self.alpha)))
                    
        else:
            self.corrective_velocity_factor = Function(self.P1DG).interpolate(Constant(1.0))
        
        self.z_n = Function(P1)
        self.z_n1 = Function(P1)
        self.v = TestFunction(P1)
        self.old_bathymetry_2d = interpolate(self.bathymetry, P1)
        
        # define bed gradient
        self.dzdx = interpolate(self.old_bathymetry_2d.dx(0), P1)
        self.dzdy = interpolate(self.old_bathymetry_2d.dx(1), P1)
        
    def set_up_bedload(self, mesh):   
        P1 = FunctionSpace(mesh, "CG", 1)
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)

        #calculate angle of flow
        self.calfa = interpolate(self.horizontal_velocity/sqrt(self.unorm), P1)
        self.salfa = interpolate(self.vertical_velocity/sqrt(self.unorm), P1)
        self.div_function = interpolate(as_vector((self.calfa, self.salfa)), P1_vec)
        
        self.beta = 1.3
        
        self.surbeta2 = Constant(1/1.5)
        self.cparam = Constant((2650-1000)*9.81*self.average_size*(self.surbeta2**2))
        
        if self.slope_eff:    
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            self.slopecoef = interpolate(1 + self.beta*(self.dzdx*self.calfa + self.dzdy*self.salfa), P1)
        else:
            self.slopecoef = interpolate(Constant(1.0), P1)
            
        if self.angle_correction == True:
            # slope effect angle correction due to gravity
            self.tt1 = interpolate(conditional(1000*0.5*self.qfc*self.unorm > 10**(-10), sqrt(self.cparam/(1000*0.5*self.qfc*self.unorm)), sqrt(self.cparam/(10**(-10)))), P1)
            # add on a factor of the bed gradient to the normal
            self.aa = interpolate(self.salfa + self.tt1*self.dzdy, P1)
            self.bb = interpolate(self.calfa + self.tt1*self.dzdx, P1)
            self.norm = interpolate(conditional(sqrt(self.aa**2 + self.bb**2) > 10**(-10), sqrt(self.aa**2 + self.bb**2),10**(-10)), P1)
            self.calfamod = interpolate(self.bb/self.norm, P1)
            self.salfamod = interpolate(self.aa/self.norm, P1)            

        # implement meyer-peter-muller bedload transport formula
        self.thetaprime = interpolate(self.mu*(1000*0.5*self.qfc*self.unorm)/((2650-1000)*9.81*self.average_size), P1)

        # if velocity above a certain critical value then transport occurs
        self.phi = interpolate(conditional(self.thetaprime < self.thetacr, 0, 8*(self.thetaprime-self.thetacr)**1.5), P1)
        
        self.z_n = Function(P1)
        self.z_n1 = Function(P1)
        self.v = TestFunction(P1)
        self.n = FacetNormal(mesh)
        self.old_bathymetry_2d = Function(P1)
        

    def update_key_hydro(self, solver_obj):
        
        self.old_bathymetry_2d.assign(solver_obj.fields.bathymetry_2d)
        
        
        
        self.z_n.interpolate(self.old_bathymetry_2d) 
        
        self.uv1, self.eta = solver_obj.fields.solution_2d.split()
        self.u_cg.interpolate(self.uv1)
        self.elev_cg.interpolate(self.eta)

        # calculate gradient of bed (noting bathymetry is -bed)
        self.dzdx.interpolate(self.old_bathymetry_2d.dx(0))
        self.dzdy.interpolate(self.old_bathymetry_2d.dx(1))

        self.horizontal_velocity.interpolate(self.u_cg[0])
        self.vertical_velocity.interpolate(self.u_cg[1])
            
        # Update depth
        if self.wetting_and_drying:
            bathymetry_displacement = solver_obj.eq_sw.depth.wd_bathymetry_displacement
            self.depth.interpolate(self.elev_cg + bathymetry_displacement(self.eta) + self.bathymetry)
        else:
            self.depth.interpolate(self.elev_cg + self.bathymetry)

        self.hc = conditional(self.depth > 0.001, self.depth, 0.001)
        self.aux = conditional(11.036*self.hc/self.ks > 1.001, 11.036*self.hc/self.ks, 1.001)
        self.qfc = interpolate(2/(ln(self.aux)/0.4)**2, P1DG)
        
        # calculate skin friction coefficient
        self.cfactor.interpolate(self.get_cfactor())

        self.quadratic_drag_coefficient.project(self.get_cfactor())        
        
        # mu - ratio between skin friction and normal friction
        self.mu.assign(conditional(self.qfc > 0, self.cfactor/self.qfc, 0))
            
        # bed shear stress
        self.unorm.interpolate((self.horizontal_velocity**2) + (self.vertical_velocity**2))
        self.TOB.interpolate(1000*0.5*self.qfc*self.unorm)       
        
        self.f = (((1-self.porosity)*(self.z_n1 - self.z_n)/(self.dt*self.morfac))*self.v)*dx
        
    def update_suspended(self, solver_obj):
        P1DG = solver_obj.function_spaces.P1DG_2d

        self.B.interpolate(conditional(self.a > self.depth, 1, self.a/self.depth))
        self.ustar.interpolate(sqrt(0.5*self.qfc*self.unorm))
        self.exp1.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), conditional((self.settling_velocity/(0.4*self.ustar)) -1 > 3, 3, (self.settling_velocity/(0.4*self.ustar))-1), 0))
        self.coefftest.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), self.B*(1-self.B**self.exp1)/self.exp1, -self.B*ln(self.B)))
        self.coeff.interpolate(conditional(self.coefftest>0, 1/self.coefftest, 0))
        
        # erosion flux - van rijn
        self.s0.assign((conditional(1000*0.5*self.qfc*self.unorm*self.mu > 0, 1000*0.5*self.qfc*self.unorm*self.mu, 0) - self.taucr)/self.taucr)
        self.ceq.interpolate(0.015*(self.average_size/self.a) * ((conditional(self.s0 < 0, 0, self.s0))**(1.5))/(self.dstar**0.3))
        self.tracer_init_value.assign(self.ceq.at([0,0])/self.coeff.at([0,0]))


        self.source.interpolate(self.set_source_tracer(P1DG, solver_obj))
        
        
        self.qbsourcedepth.interpolate(self.source*self.depth)
        
        if self.convective_vel_flag:
            
            # correction factor to advection velocity in sediment concentration equation
            self.Bconv.interpolate(conditional(self.depth > 1.1*self.ksp, self.ksp/self.depth, self.ksp/(1.1*self.ksp)))
            self.Aconv.interpolate(conditional(self.depth > 1.1* self.a, self.a/self.depth, self.a/(1.1*self.a)))
                    
            # take max of value calculated either by ksp or depth
            self.Amax.assign(conditional(self.Aconv > self.Bconv, self.Aconv, self.Bconv))

            self.r1conv.assign(1 - (1/0.4)*conditional(self.settling_velocity/self.ustar < 1, self.settling_velocity/self.ustar, 1))

            self.Ione.assign(conditional(self.r1conv > 10**(-8), (1 - self.Amax**self.r1conv)/self.r1conv, conditional(self.r1conv < - 10**(-8), (1 - self.Amax**self.r1conv)/self.r1conv, ln(self.Amax))))

            self.Itwo.assign(conditional(self.r1conv > 10**(-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, conditional(self.r1conv < - 10**(-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, -0.5*ln(self.Amax)**2)))

            self.alpha.assign(-(self.Itwo - (ln(self.Amax) - ln(30))*self.Ione)/(self.Ione * ((ln(self.Amax) - ln(30)) + 1)))

            # final correction factor
            self.corrective_velocity_factor.assign(conditional(conditional(self.alpha > 1, 1, self.alpha) < 0, 0, conditional(self.alpha > 1, 1, self.alpha)))

        self.f += - (self.qbsourcedepth * self.v)*dx

    def update_bedload(self, solver_obj):

        # calculate angle of flow
        self.calfa.interpolate(self.horizontal_velocity/sqrt(self.unorm))
        self.salfa.interpolate(self.vertical_velocity/sqrt(self.unorm))
        self.div_function.interpolate(as_vector((self.calfa, self.salfa)))
        
        if self.slope_eff:    
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            # we use z_n1 and equals so that we can use an implicit method in Exner
            self.slopecoef = (1 + self.beta*(self.z_n1.dx(0)*self.calfa + self.z_n1.dx(1)*self.salfa))
        else:
            self.slopecoef = Constant(1.0)  
            
        if self.angle_correction == True:
            # slope effect angle correction due to gravity
            self.tt1.interpolate(conditional(1000*0.5*self.qfc*self.unorm > 10**(-10), sqrt(self.cparam/(1000*0.5*self.qfc*self.unorm)), sqrt(self.cparam/(10**(-10)))))
            # add on a factor of the bed gradient to the normal
            self.aa.assign(self.salfa + self.tt1*self.dzdy)
            self.bb.assign(self.calfa + self.tt1*self.dzdx)
            self.norm.assign(conditional(sqrt(self.aa**2 + self.bb**2) > 10**(-10), sqrt(self.aa**2 + self.bb**2),10**(-10)))
            # we use z_n1 and equals so that we can use an implicit method in Exner
            self.calfamod = (self.calfa + (self.tt1*self.z_n1.dx(0)))/self.norm
            self.salfamod = (self.salfa + (self.tt1*self.z_n1.dx(1)))/self.norm              
            
        # implement meyer-peter-muller bedload transport formula
        self.thetaprime.interpolate(self.mu*(1000*0.5*self.qfc*self.unorm)/((2650-1000)*9.81*self.average_size))

        # if velocity above a certain critical value then transport occurs
        self.phi.assign(conditional(self.thetaprime < self.thetacr, 0, 8*(self.thetaprime-self.thetacr)**1.5))
                        
        # bedload transport flux with magnitude correction
        self.qb_total = self.slopecoef*self.phi*sqrt(self.g*(2650/1000 - 1)*self.average_size**3)            
        
        # formulate bedload transport flux with correct angle depending on corrections implemented
        if self.angle_correction == True:
            self.qbx = self.qb_total*self.calfamod
            self.qby = self.qb_total*self.salfamod                     
        else:
            self.qbx = self.qb_total*self.calfa
            self.qby = self.qb_total*self.salfa                            
                    
        # add bedload transport to exner equation
        self.f += -(self.v*((self.qbx*self.n[0]) + (self.qby*self.n[1])))*ds(1) -(self.v*((self.qbx*self.n[0]) + (self.qby*self.n[1])))*ds(2) + (self.qbx*(self.v.dx(0)) + self.qby*(self.v.dx(1)))*dx                
