from thetis import *
from thetis.configuration import *

from adapt_utils.swe.tsunami.options import TsunamiOptions

__all__ = ["TracerOptions"]

class TracerOptions(TsunamiOptions):
    """
    Parameter class for general morphological problems.
    """
    
    def __init__(self, **kwargs):
        import ipdb; ipdb.set_trace()
        super(TracerOptions, self).__init__(**kwargs)    
    
    def set_up_suspended(self):
        
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

        self.u_cg = Function(self.vector_cg).project(self.uv_init)
        self.horizontal_velocity = Function(self.V).project(self.u_cg[0])
        self.vertical_velocity = Function(self.V).project(self.u_cg[1])
        self.elev_cg = Function(self.V).project(self.eta_init)

    
        self.unorm = Function(self.P1DG).project((self.horizontal_velocity**2)+ (self.vertical_velocity**2))

        self.hc = Function(self.P1DG).project(conditional(self.depth > 0.001, self.depth, 0.001))
        self.aux = Function(self.P1DG).project(conditional(11.036*hc/self.ks > 1.001, 11.036*hc/self.ks, 1.001))
        self.qfc = Function(self.P1DG).project(2/(ln(self.aux)/0.4)**2)
        
        self.TOB = Function(self.V).project(1000*0.5*self.qfc*self.unorm)
        
        
        # skin friction coefficient
        
        self.cfactor = Function(self.P1DG).project(self.get_cfactor())
        # mu - ratio between skin friction and normal friction
        self.mu = Function(self.P1DG).interpolate(conditional(self.qfc > 0, self.cfactor/self.qfc, 0))
        
        
        self.a = (self.ks)/2
        self.B = Function(self.P1DG).interpolate(conditional(self.a > self.depth, 1, self.a/self.depth))
        self.ustar = Function(self.P1DG).interpolate(sqrt(0.5*self.qfc*self.unorm))
        self.exp1 = Function(self.P1DG).interpolate(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), conditional((self.settling_velocity/(0.4*self.ustar)) -1 > 3, 3, (self.settling_velocity/(0.4*self.ustar))-1), 0))
        self.coefftest = Function(self.P1DG).interpolate(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), self.B*(1-self.B**self.exp1)/self.exp1, -self.B*ln(self.B)))
        self.coeff = Function(self.P1DG).interpolate(conditional(self.coefftest>0, 1/self.coefftest, 0))
        
        
        # erosion flux - for vanrijn
        self.s0 = Function(self.P1DG).interpolate((conditional(1000*0.5*self.qfc*self.unorm*self.mu > 0, 1000*0.5*self.qfc*self.unorm*self.mu, 0) - self.taucr)/self.taucr)
        self.ceq = Function(self.P1DG).interpolate(0.015*(self.average_size/self.a) * ((conditional(self.s0 < 0, 0, self.s0))**(1.5))/(self.dstar**0.3))
        
        self.testtracer = Function(self.P1DG).project(self.tracer_init_value)
        self.source = self.set_source_tracer(self.P1DG, solver_obj = None, init = True)   
        self.qbsourcedepth = Function(self.V).project(self.source * self.depth)
        
        self.z_n = Function(self.V)
        self.z_n1 = Function(self.V)
        self.v = TestFunction(self.V)
        self.old_bathymetry_2d = Function(self.V)
        
    def update_suspended(self, solver_obj):
        
        self.old_bathymetry_2d.assign(solver_obj.fields.bathymetry_2d)
        
        self.z_n.assign(self.old_bathymetry_2d)
        
        # mu - ratio between skin friction and normal friction
        self.mu.assign(conditional(self.qfc > 0, self.cfactor/self.qfc, 0))
            
        # bed shear stress
        self.unorm.interpolate((self.horizontal_velocity**2) + (self.vertical_velocity**2))
        self.TOB.interpolate(1000*0.5*self.qfc*self.unorm)
        
        self.B.interpolate(conditional(self.a > self.depth, 1, self.a/self.depth))
        self.ustar.interpolate(sqrt(0.5*self.qfc*self.unorm))
        self.exp1.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), conditional((self.settling_velocity/(0.4*self.ustar)) -1 > 3, 3, (self.settling_velocity/(0.4*self.ustar))-1), 0))
        self.coefftest.assign(conditional((conditional((self.settling_velocity/(0.4*self.ustar)) - 1 > 0, (self.settling_velocity/(0.4*self.ustar)) -1, -(self.settling_velocity/(0.4*self.ustar)) + 1)) > 10**(-4), self.B*(1-self.B**self.exp1)/self.exp1, -self.B*ln(self.B)))
        self.coeff.assign(conditional(self.coefftest>0, 1/self.coefftest, 0))
        
        # erosion flux - van rijn
        self.s0.assign((conditional(1000*0.5*self.qfc*self.unorm*self.mu > 0, 1000*0.5*self.qfc*self.unorm*self.mu, 0) - self.taucr)/self.taucr)
        self.ceq.assign(0.015*(self.average_size/self.a) * ((conditional(self.s0 < 0, 0, self.s0))**(1.5))/(self.dstar**0.3))
        
        self.source.project(self.set_source_tracer(self.eta.function_space(), solver_obj))
        
        f = (((1-self.porosity)*(self.z_n1 - self.z_n)/(self.dt*self.morfac))*self.v)*dx
        
        self.qbsourcedepth.interpolate(self.source*self.depth)
        
        f += - (self.qbsourcedepth * self.v)*dx
        
        solve(f==0, self.z_n1)
        
        solver_obj.fields.bathymetry_2d.assign(self.z_n1)
