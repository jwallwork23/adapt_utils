from thetis import *
from thetis.configuration import *

from adapt_utils.swe.tsunami.options import TsunamiOptions

__all__ = ["TracerOptions"]

class TracerOptions(TsunamiOptions):
    """
    Parameter class for general morphological problems.
    """
    
    def __init__(self, **kwargs):
        super(TracerOptions, self).__init__(**kwargs)    
    
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

        self.bathymetry_file = File(self.di + "/bathy.pvd")
            
        self.bathymetry_file.write(self.bathymetry)
        
        #import ipdb; ipdb.set_trace()
                     

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
        
        

        self.tracer_file = File(self.di + "/tracery.pvd")
            
        self.tracer_file.write(self.tracer_init)
        
        
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
        self.old_bathymetry_2d = Function(self.P1)

        
    def update_suspended(self, solver_obj):
        
        #self.bathymetry.interpolate(self.set_bathymetry(self.P1))
        
        self.old_bathymetry_2d.assign(self.bathymetry)
        
        self.z_n.interpolate(self.old_bathymetry_2d)
        
        # mu - ratio between skin friction and normal friction
        self.mu.assign(conditional(self.qfc > 0, self.cfactor/self.qfc, 0))
            
        # bed shear stress
        self.unorm.interpolate((self.horizontal_velocity**2) + (self.vertical_velocity**2))
        self.TOB.interpolate(1000*0.5*self.qfc*self.unorm)
        
        self.B.interpolate(conditional(self.a > self.depth, 1, self.a/self.depth))
        self.ustar.interpolate(sqrt(0.5*self.qfc*self.unorm))
        self.exp1.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), conditional((self.settling_velocity/(0.4*self.ustar)) -1 > 3, 3, (self.settling_velocity/(0.4*self.ustar))-1), 0))
        self.coefftest.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), self.B*(1-self.B**self.exp1)/self.exp1, -self.B*ln(self.B)))
        self.coeff.interpolate(conditional(self.coefftest>0, 1/self.coefftest, 0))
        
        # erosion flux - van rijn
        self.s0.assign((conditional(1000*0.5*self.qfc*self.unorm*self.mu > 0, 1000*0.5*self.qfc*self.unorm*self.mu, 0) - self.taucr)/self.taucr)
        self.ceq.interpolate(0.015*(self.average_size/self.a) * ((conditional(self.s0 < 0, 0, self.s0))**(1.5))/(self.dstar**0.3))
        self.tracer_init_value.assign(self.ceq.at([0,0])/self.coeff.at([0,0]))
        print('tracer')
        print(solver_obj.bnd_functions['tracer'][1]['value'].dat.data[:])
        print(solver_obj.fields.tracer_2d.at([0,0]))
        self.source.interpolate(self.set_source_tracer(self.P1DG, solver_obj))
        
        f = (((1-self.porosity)*(self.z_n1 - self.z_n)/(self.dt*self.morfac))*self.v)*dx
        
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
        
        f += - (self.qbsourcedepth * self.v)*dx
        
        solve(f==0, self.z_n1)
        
        self.bathymetry.assign(self.z_n1)
        solver_obj.fields.bathymetry_2d.assign(self.z_n1)
        print(max(self.bathymetry.dat.data[:]))