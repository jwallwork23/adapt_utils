from thetis import *
from thetis.configuration import *

from adapt_utils.swe.tsunami.options import TsunamiOptions

__all__ = ["MorphOptions"]

class MorphOptions(TsunamiOptions):
    """
    Parameter class for general morphological problems.
    """
    
    def __init__(self, **kwargs):
        super(MorphOptions, self).__init__(**kwargs)    
    
    def set_up_suspended(self, mesh): 
        

        
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

        self.uv_d = Function(self.P1_vec_dg).project(self.uv_d)
        self.eta_d = Function(self.P1DG).project(self.eta_d)
        
        self.u_cg = Function(self.P1_vec).project(self.uv_d)
        self.horizontal_velocity = Function(self.P1).project(self.u_cg[0])
        self.vertical_velocity = Function(self.P1).project(self.u_cg[1])
        self.elev_cg = Function(self.P1).project(self.eta_d)
        
        
        if self.t_old.dat.data[:] == 0.0:
            self.set_bathymetry(self.P1)
        else:
            self.bathymetry = Function(self.P1).project(self.bathymetry)

        self.depth = Function(self.P1).project(self.elev_cg + self.bathymetry)
    
        self.unorm = Function(self.P1DG).project((self.horizontal_velocity**2)+ (self.vertical_velocity**2))

        self.hc = Function(self.P1DG).project(conditional(self.depth > 0.001, self.depth, 0.001))
        self.aux = Function(self.P1DG).project(conditional(11.036*self.hc/self.ks > 1.001, 11.036*self.hc/self.ks, 1.001))
        self.qfc = Function(self.P1DG).project(2/(ln(self.aux)/0.4)**2)
        
        self.TOB = Function(self.P1).project(1000*0.5*self.qfc*self.unorm)
        
        
        # skin friction coefficient
        
        self.cfactor = Function(self.P1DG).project(self.get_cfactor())
        # mu - ratio between skin friction and normal friction
        self.mu = Function(self.P1DG).project(conditional(self.qfc > 0, self.cfactor/self.qfc, 0))
        
        
        self.a = (self.ks)/2
        self.B = Function(self.P1DG).project(conditional(self.a > self.depth, 1, self.a/self.depth))
        self.ustar = Function(self.P1DG).project(sqrt(0.5*self.qfc*self.unorm))
        self.exp1 = Function(self.P1DG).project(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), conditional((self.settling_velocity/(0.4*self.ustar)) -1 > 3, 3, (self.settling_velocity/(0.4*self.ustar))-1), 0))
        self.coefftest = Function(self.P1DG).project(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), self.B*(1-self.B**self.exp1)/self.exp1, -self.B*ln(self.B)))
        self.coeff = Function(self.P1DG).project(conditional(self.coefftest>0, 1/self.coefftest, 0))
        
        
        # erosion flux - for vanrijn
        self.s0 = Function(self.P1DG).project((conditional(1000*0.5*self.qfc*self.unorm*self.mu > 0, 1000*0.5*self.qfc*self.unorm*self.mu, 0) - self.taucr)/self.taucr)
        self.ceq = Function(self.P1DG).project(0.015*(self.average_size/self.a) * ((conditional(self.s0 < 0, 0, self.s0))**(1.5))/(self.dstar**0.3))
        
        self.tracer_init = Function(self.P1DG).project(self.ceq/self.coeff)
        
        
        self.tracer_init_value = Constant(self.ceq.at([0,0])/self.coeff.at([0,0]))
        self.source = Function(self.P1DG).project(self.set_source_tracer(self.P1DG, solver_obj = None, init = True, t_old = self.t_old)) 
        self.qbsourcedepth = Function(self.P1).project(self.source * self.depth)
        
        if self.convective_vel_flag:
            # correction factor to advection velocity in sediment concentration equation

            self.Bconv = Function(self.P1DG).interpolate(conditional(self.depth > 1.1*self.ksp, self.ksp/self.depth, self.ksp/(1.1*self.ksp)))
            self.Aconv = Function(self.P1DG).interpolate(conditional(self.depth > 1.1* self.a, self.a/self.depth, self.a/(1.1*self.a)))
                    
            # take max of value calculated either by ksp or depth
            self.Amax = Function(self.P1DG).interpolate(conditional(self.Aconv > self.Bconv, self.Aconv, self.Bconv))

            self.r1conv = Function(self.P1DG).interpolate(1 - (1/0.4)*conditional(self.settling_velocity/self.ustar < 1, self.settling_velocity/self.ustar, 1))

            self.Ione = Function(self.P1DG).interpolate(conditional(self.r1conv > 10**(-8), (1 - self.Amax**self.r1conv)/self.r1conv, conditional(self.r1conv < - 10**(-8), (1 - self.Amax**self.r1conv)/self.r1conv, ln(self.Amax))))

            self.Itwo = Function(self.P1DG).interpolate(conditional(self.r1conv > 10**(-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, conditional(self.r1conv < - 10**(-8), -(self.Ione + (ln(self.Amax)*(self.Amax**self.r1conv)))/self.r1conv, -0.5*ln(self.Amax)**2)))

            self.alpha = Function(self.P1DG).interpolate(-(self.Itwo - (ln(self.Amax) - ln(30))*self.Ione)/(self.Ione * ((ln(self.Amax) - ln(30)) + 1)))

            # final correction factor
            self.alphatest2 = Function(self.P1DG).interpolate(conditional(conditional(self.alpha > 1, 1, self.alpha) < 0, 0, conditional(self.alpha > 1, 1, self.alpha)))
                    
            # multiply correction factor by velocity and insert back into sediment concentration equation
            self.corrective_velocity = Function(self.P1_vec).interpolate(self.alphatest2 * self.uv_d)
        else:
            self.corrective_velocity = Function(self.P1_vec).interpolate(self.uv_d)
        
        self.z_n = Function(self.P1)
        self.z_n1 = Function(self.P1)
        self.v = TestFunction(self.P1)
        self.old_bathymetry_2d = Function(self.P1).interpolate(self.bathymetry)
        
        # define bed gradient
        self.dzdx = Function(self.P1).interpolate(self.old_bathymetry_2d.dx(0))
        self.dzdy = Function(self.P1).interpolate(self.old_bathymetry_2d.dx(1))
        
    def set_up_bedload(self, mesh):   

        #calculate angle of flow
        self.calfa = Function(self.P1).interpolate(self.horizontal_velocity/sqrt(self.unorm))
        self.salfa = Function(self.P1).interpolate(self.vertical_velocity/sqrt(self.unorm))
        self.div_function = Function(self.P1_vec).interpolate(as_vector((self.calfa, self.salfa)))
        
        self.beta = 1.3
        
        self.surbeta2 = Constant(1/1.5)
        self.cparam = Constant((2650-1000)*9.81*self.average_size*(self.surbeta2**2))
        
        if self.slope_eff:    
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            self.slopecoef = Function(self.P1).interpolate(1 + self.beta*(self.dzdx*self.calfa + self.dzdy*self.salfa))
        else:
            self.slopecoef = Function(self.P1).interpolate(Constant(1.0))
            
        if self.angle_correction == True:
            # slope effect angle correction due to gravity
            self.tt1 = Function(self.P1).interpolate(conditional(1000*0.5*self.qfc*self.unorm > 10**(-10), sqrt(self.cparam/(1000*0.5*self.qfc*self.unorm)), sqrt(self.cparam/(10**(-10)))))
            # add on a factor of the bed gradient to the normal
            self.aa = Function(self.P1).interpolate(self.salfa + self.tt1*self.dzdy)
            self.bb = Function(self.P1).interpolate(self.calfa + self.tt1*self.dzdx)
            self.norm = Function(self.P1).interpolate(conditional(sqrt(self.aa**2 + self.bb**2) > 10**(-10), sqrt(self.aa**2 + self.bb**2),10**(-10)))
            self.calfamod = Function(self.P1).interpolate(self.bb/self.norm)
            self.salfamod = Function(self.P1).interpolate(self.aa/self.norm)            

        # implement meyer-peter-muller bedload transport formula
        self.thetaprime = Function(self.P1).interpolate(self.mu*(1000*0.5*self.qfc*self.unorm)/((2650-1000)*9.81*self.average_size))

        # if velocity above a certain critical value then transport occurs
        self.phi = Function(self.P1).interpolate(conditional(self.thetaprime < self.thetacr, 0, 8*(self.thetaprime-self.thetacr)**1.5))
        
        self.z_n = Function(self.P1)
        self.z_n1 = Function(self.P1)
        self.v = TestFunction(self.P1)
        self.n = FacetNormal(self.P1.mesh())
        self.old_bathymetry_2d = Function(self.P1)        
        

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
            bathymetry_displacement =   solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
            self.depth.interpolate(self.elev_cg + bathymetry_displacement(self.eta) + self.bathymetry)
        else:
            self.depth.interpolate(self.elev_cg + self.bathymetry)

            
        self.hc.interpolate(conditional(self.depth > 0.001, self.depth, 0.001))
        self.aux.interpolate(conditional(11.036*self.hc/self.ks > 1.001, 11.036*self.hc/self.ks, 1.001))
        self.qfc.interpolate(2/(ln(self.aux)/0.4)**2)
        
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

        self.B.interpolate(conditional(self.a > self.depth, 1, self.a/self.depth))
        self.ustar.interpolate(sqrt(0.5*self.qfc*self.unorm))
        self.exp1.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), conditional((self.settling_velocity/(0.4*self.ustar)) -1 > 3, 3, (self.settling_velocity/(0.4*self.ustar))-1), 0))
        self.coefftest.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), self.B*(1-self.B**self.exp1)/self.exp1, -self.B*ln(self.B)))
        self.coeff.interpolate(conditional(self.coefftest>0, 1/self.coefftest, 0))
        
        # erosion flux - van rijn
        self.s0.assign((conditional(1000*0.5*self.qfc*self.unorm*self.mu > 0, 1000*0.5*self.qfc*self.unorm*self.mu, 0) - self.taucr)/self.taucr)
        self.ceq.interpolate(0.015*(self.average_size/self.a) * ((conditional(self.s0 < 0, 0, self.s0))**(1.5))/(self.dstar**0.3))
        self.tracer_init_value.assign(self.ceq.at([0,0])/self.coeff.at([0,0]))


        self.source.interpolate(self.set_source_tracer(self.P1DG, solver_obj))
        
        
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
            self.alphatest2.assign(conditional(conditional(self.alpha > 1, 1, self.alpha) < 0, 0, conditional(self.alpha > 1, 1, self.alpha)))
                    
            # multiply correction factor by velocity and insert back into sediment concentration equation
            self.corrective_velocity.interpolate(self.alphatest2 * self.uv1)            

        else:
            self.corrective_velocity.interpolate(self.uv1)
        
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
