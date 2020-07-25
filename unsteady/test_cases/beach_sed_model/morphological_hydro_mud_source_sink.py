#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:05:02 2019

@author: mc4117
"""

import thetis as th
import time
import datetime
import numpy as np
import pandas as pd
import firedrake as fire
from firedrake.petsc import PETSc
import math
import os
import callback_cons_tracer as call


def hydrodynamics_only(boundary_conditions_fn, mesh2d, bathymetry_2d, uv_init,\
                       elev_init, average_size, dt, t_end, wetting_and_drying = False,\
                       wetting_alpha = 0.1, friction = 'nikuradse', friction_coef = 0,\
                       fluc_bcs = False, viscosity = 10**(-6), sponge_viscosity = th.Constant(1.0)):
    """
    Sets up a simulation with only hydrodynamics until a quasi-steady state when it can be used as an initial
    condition for the full morphological model. We update the bed friction at each time step.
    The actual run of the model are done outside the function
    
    Inputs:
    boundary_consditions_fn - function defining boundary conditions for problem 
    mesh2d - define mesh working on
    bathymetry2d - define bathymetry of problem
    uv_init - initial velocity of problem
    elev_init - initial elevation of problem
    average_size - average sediment size
    dt - timestep
    t_end - end time
    wetting_and_drying - wetting and drying switch
    wetting_alpha - wetting and drying parameter
    friction - choice of friction formulation - nikuradse and manning
    friction_coef - value of friction coefficient used in manning    
    viscosity - viscosity of hydrodynamics. The default value should be 10**(-6)
    
    Outputs:
    solver_obj - solver which we need to run to solve the problem
    update_forcings_hydrodynamics - function defining the updates to the model performed at each timestep
    outputdir - directory of outputs
    """
    def update_forcings_hydrodynamics(t_new):
        # update boundary conditions if have fluctuating conditions
        if fluc_bcs == True:
            in_fn, out_fn = boundary_conditions_fn(bathymetry_2d, t_new = t_new, state = 'update')
            for j in range(len(in_fn)):
                exec('constant_in' + str(j) + '.assign(' + str(in_fn[j]) + ')')

            for k in range(len(out_fn)):	
                exec('constant_out' + str(k) + '.assign(' + str(out_fn[k]) + ')')
        # update bed friction
        if friction == 'nikuradse':
            uv1, elev1 = solver_obj.fields.solution_2d.split()
            
            if wetting_and_drying == True:
                wd_bath_displacement = solver_obj.depth.wd_bathymetry_displacement
                depth.project(elev1 + wd_bath_displacement(elev1) + bathymetry_2d)        
            else:
                depth.interpolate(elev1 + bathymetry_2d)        
   
        # calculate skin friction coefficient
        cfactor.interpolate(2*(0.4**2)/((th.ln(11.036*depth/(ksp)))**2))

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st

    # export interval in seconds
    if t_end < 40:
        t_export = 0.1
    else:
        t_export = np.round(t_end/40,0)
        

    th.print_output('Exporting to '+outputdir)

    # define function spaces
    V = th.FunctionSpace(mesh2d, 'CG', 1)
    P1_2d = th.FunctionSpace(mesh2d, 'DG', 1)

    # define parameters
    ksp = th.Constant(3*average_size)

    # define depth
    if wetting_and_drying:
        H = th.Function(V).project(elev_init + bathymetry_2d)
        depth = th.Function(V).project(H + (0.5 * (th.sqrt(H ** 2 + wetting_alpha ** 2) - H)))
    else:
        depth = th.Function(V).project(elev_init + bathymetry_2d)

    # define bed friction
    cfactor = th.Function(P1_2d).interpolate(2*(0.4**2)/((th.ln(11.036*depth/(ksp)))**2))
    
    # set up solver 
    solver_obj = th.solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir

    options.check_volume_conservation_2d = True

    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.solve_tracer = False
    options.use_lax_friedrichs_tracer = False
    if friction == 'nikuradse':
        options.quadratic_drag_coefficient = cfactor
    elif friction == 'manning':
        if friction_coef == 0:
            friction_coef = 0.02
        options.manning_drag_coefficient = th.Constant(friction_coef)
    else:
        print('Undefined friction')
    options.horizontal_viscosity = th.Function(P1_2d).interpolate(sponge_viscosity*th.Constant(viscosity))

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.use_wetting_and_drying = wetting_and_drying
    options.wetting_and_drying_alpha = th.Function(V).interpolate(th.Constant(wetting_alpha))
    options.norm_smoother = th.Constant(1.0) #wetting_alpha)

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    # set boundary conditions

    swe_bnd, left_bnd_id, right_bnd_id, in_constant, out_constant, left_string, right_string = boundary_conditions_fn(bathymetry_2d, flag = 'hydro')
    
    for j in range(len(in_constant)):
        exec('constant_in' + str(j) + ' = th.Constant(' + str(in_constant[j]) + ')', globals())

    str1 = '{'
    if len(left_string) > 0:
        for i in range(len(left_string)):
            str1 += "'"+ str(left_string[i]) + "': constant_in" + str(i) + ","
        str1 = str1[0:len(str1)-1] + "}"
        exec('swe_bnd[left_bnd_id] = ' + str1)

    for k in range(len(out_constant)):
        exec('constant_out' + str(k) + '= th.Constant(' + str(out_constant[k]) + ')', globals())

    str2 = '{'
    if len(right_string) > 0:
        for i in range(len(right_string)):
            str2 += "'"+ str(right_string[i]) + "': constant_out" + str(i) + ","
        str2 = str2[0:len(str2)-1] + "}"
        exec('swe_bnd[right_bnd_id] = ' + str2)
   
    solver_obj.bnd_functions['shallow_water'] = swe_bnd
   
    print(swe_bnd)
    solver_obj.assign_initial_conditions(uv = uv_init, elev= elev_init)
    
    return solver_obj, update_forcings_hydrodynamics, outputdir


def morphological(boundary_conditions_fn, morfac, morfac_transport, suspendedload, convectivevel, bedload, angle_correction, slope_eff, seccurrent, \
                 mesh2d, bathymetry_2d, input_dir, viscosity_hydro, ks, average_size, dt, final_time, wetting_and_drying = False, wetting_alpha = 0.1, rhos = 2650, sponge_viscosity = th.Constant(1.0),\
                 beta_fn = 1.3, surbeta2_fn = 1/1.5, alpha_secc_fn = 0.75, angle_fn = 35, mesh_step_size = 0.2, update_bedlevel = True, cons_tracer = False,\
                 friction = 'nikuradse', friction_coef = 0, d90 = 0, fluc_bcs = False, bed_form = 'meyer', sus_form = 'vanrijn', diffusivity = 0.15, tracer_init = None, depth_integrated = False):
    """
    Set up a full morphological model simulation using as an initial condition the results of a hydrodynamic only model.    
    
    Inputs:
    boundary_consditions_fn - function defining boundary conditions for problem 
    morfac - morphological scale factor 
    morfac_transport - switch to turn on morphological component
    suspendedload - switch to turn on suspended sediment transport
    convectivevel - switch on convective velocity correction factor in sediment concentration equation
    bedload - switch to turn on bedload transport
    angle_correction - switch on slope effect angle correction
    slope_eff - switch on slope effect magnitude correction
    seccurrent - switch on secondary current for helical flow effect
    sediment_slide - switch on sediment slide mechanism to prevent steep slopes
    mesh2d - define mesh working on
    bathymetry2d - define bathymetry of problem
    input_dir - folder containing results of hydrodynamics model which are used as initial conditions here
    viscosity_hydro - viscosity value in hydrodynamic equations
    ks - bottom friction coefficient for quadratic drag coefficient
    average_size - average sediment size
    dt - timestep
    final_time - end time    
    wetting_and_drying - wetting and drying switch
    wetting_alpha - wetting and drying parameter
    rhos - sediment density    
    beta_fn - magnitude slope effect parameter
    surbeta2_fn - angle correction slope effect parameter
    alpha_secc_fn - secondary current parameter
    angle_fn - maximum angle allowed by sediment slide mechanism
    mesh_step_size - NOT for defining mesh but to be used in the sediment slide mechanism
    update_bedlevel - switch to update bathymetry
    cons_tracer - switch to conservative tracer
    friction - choice of friction formulation - nikuradse and manning
    friction_coef - value of friction coefficient used in manning
    d90 - sediment size which 90% of the sediment are below
    fluc_bcs - switch on fluctuating boundary conditions
    bed_form - choice of bedload formula between 'meyer' (meyer-peter-muller) and 'soulsby' (soulsby-van-rijn)
    sus_form - choice of suspended load formula between 'vanrijn' (van Rijn 1984) and 'soulsby' (soulsby-van-rijn)
    diffusivity - value of diffusivity coefficient
    tracer_init - initial tracer value
    depth_integrated - switch for depth-integrated sources
    
    Outputs:
    solver_obj - solver which we need to run to solve the problem
    update_forcings_hydrodynamics - function defining the updates to the model performed at each timestep
    diff_bathy - bedlevel evolution
    diff_bathy_file - bedlevel evolution file
    """
    t_list = []    
    
    def update_forcings_tracer(t_new):

        # update bathymetry
        old_bathymetry_2d.assign(bathymetry_2d)
        
        # extract new elevation and velocity and project onto CG space
        uv1, elev1 = solver_obj.fields.solution_2d.split()
        uv_cg.project(uv1)
        
        if wetting_and_drying == True:
            wd_bath_displacement = solver_obj.depth.wd_bathymetry_displacement
            depth.project(elev1 + wd_bath_displacement(elev1) + old_bathymetry_2d)  
            elev_cg.project(elev1 + wd_bath_displacement(elev1))	
        else:              
            elev_cg.project(elev1)
            depth.project(elev_cg + old_bathymetry_2d)
        
        horizontal_velocity.interpolate(uv_cg[0])
        vertical_velocity.interpolate(uv_cg[1])
      
        # update boundary conditions if have fluctuating conditions
        if fluc_bcs == True:
            in_fn, out_fn = boundary_conditions_fn(bathymetry_2d, flag = 'morpho', morfac = morfac, t_new = t_new, state = 'update')
            for j in range(len(in_fn)):
                exec('constant_in' + str(j) + '.assign(' + str(in_fn[j]) + ')')
            
            for k in range(len(out_fn)):
                exec('constant_out' + str(k) + '.assign(' + str(out_fn[k]) + ')')

        # update bedfriction 
        hc.interpolate(th.conditional(depth > 0.001, depth, 0.001))
        aux.assign(th.conditional(11.036*hc/ks > 1.001, 11.036*hc/ks, 1.001))
        qfc.assign(2/(th.ln(aux)/0.4)**2)
        
        # calculate skin friction coefficient
        hclip.interpolate(th.conditional(ksp > depth, ksp, depth))
        cfactor.interpolate(th.conditional(depth > ksp, 2*((2.5*th.ln(11.036*hclip/ksp))**(-2)), th.Constant(0.0)))
        

        if morfac_transport == True:

            # if include sediment then update_forcings is run twice but only want to update bathymetry once
            t_list.append(t_new)
            double_factor = False
            if suspendedload == True:

                if len(t_list) > 1:
                    if t_list[len(t_list)-1] == t_list[len(t_list)-2]:
                        double_factor = True       
            else:
                # if have no tracer then update_forcings is only run once so update bathymetry at each step
                double_factor = True
            
            if double_factor == True:
                z_n.assign(old_bathymetry_2d)
                    
        
                # mu - ratio between skin friction and normal friction
                mu.assign(th.conditional(qfc > 0, cfactor/qfc, 0))
            
                # bed shear stress
                unorm.interpolate((horizontal_velocity**2) + (vertical_velocity**2))
                TOB.interpolate(1000*0.5*qfc*unorm)
                
                # calculate gradient of bed (noting bathymetry is -bed)
                dzdx.interpolate(old_bathymetry_2d.dx(0))
                dzdy.interpolate(old_bathymetry_2d.dx(1))

                # initialise exner equation if not already initialised in sediment slide
                f = 0

                if suspendedload == True:
                    # source term
                
                    # deposition flux - calculating coefficient to account for stronger conc at bed
                    B.interpolate(th.conditional(a > depth, a/a, a/depth))
                    ustar.interpolate(th.sqrt(0.5*qfc*unorm))
                    exp1.assign(th.conditional((th.conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), th.conditional((settling_velocity/(0.4*ustar)) -1 > 3, 3, (settling_velocity/(0.4*ustar))-1), 0))
                    coefftest.assign(th.conditional((th.conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), B*(1-B**exp1)/exp1, -B*th.ln(B)))
                    coeff.assign(th.conditional(coefftest>0, 1/coefftest, 0))                    
        
                    if sus_form == 'vanrijn':
                        # erosion flux - above critical velocity bed is eroded
                        s0.assign((th.conditional(1000*0.5*qfc*unorm*mu > 0, 1000*0.5*qfc*unorm*mu, 0) - taucr)/taucr)
                        ceq.assign(0.015*(average_size/a) * ((th.conditional(s0 < 0, 0, s0))**(1.5))/(dstar**0.3))
                    elif sus_form == 'soulsby':
                        ucr.interpolate(0.19*(average_size**0.1)*(th.ln(4*depth/d90)/th.ln(10)))
                        s0.assign(th.conditional((th.sqrt(unorm)-ucr)**2.4 > 0, (th.sqrt(unorm)-ucr)**2.4,0))
                        ceq.interpolate(ass*s0/depth)
                    else: 
                        print('Unrecognised suspended sediment transport formula. Please choose "vanrijn" or "soulsby"')

                 
                    # calculate depth-averaged source term for sediment concentration equation

                    depo.interpolate(settling_velocity*coeff)
                    ero.interpolate(settling_velocity*ceq)

                    if cons_tracer:
                        if depth_integrated:
                            depth_int_sink.interpolate(depo/depth)
                            depth_int_source.interpolate(ero)
                        else:
                            sink.interpolate(depo/(depth**2))
                            source.interpolate(ero/depth)
                        qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d/depth)+ ero)
                    else:
                        sink.interpolate(depo/depth)
                        source.interpolate(ero/depth)
                        qbsourcedepth.interpolate(-(depo*solver_obj.fields.tracer_2d)+ ero)

                    if convectivevel == True:
                        # correction factor to advection velocity in sediment concentration equation
                
                        Bconv.interpolate(th.conditional(depth > 1.1*ksp, ksp/depth, ksp/(1.1*ksp)))
                        Aconv.interpolate(th.conditional(depth > 1.1* a, a/depth, a/(1.1*a)))
                    
                        # take max of value calculated either by ksp or depth
                        Amax.assign(th.conditional(Aconv > Bconv, Aconv, Bconv))

                        r1conv.assign(1 - (1/0.4)*th.conditional(settling_velocity/ustar < 1, settling_velocity/ustar, 1))

                        Ione.assign(th.conditional(r1conv > 10**(-8), (1 - Amax**r1conv)/r1conv, th.conditional(r1conv < - 10**(-8), (1 - Amax**r1conv)/r1conv, th.ln(Amax))))

                        Itwo.assign(th.conditional(r1conv > 10**(-8), -(Ione + (th.ln(Amax)*(Amax**r1conv)))/r1conv, th.conditional(r1conv < - 10**(-8), -(Ione + (th.ln(Amax)*(Amax**r1conv)))/r1conv, -0.5*th.ln(Amax)**2)))

                        alpha.assign(-(Itwo - (th.ln(Amax) - th.ln(30))*Ione)/(Ione * ((th.ln(Amax) - th.ln(30)) + 1)))

                        # final correction factor
                        alphatest2.assign(th.conditional(th.conditional(alpha > 1, 1, alpha) < 0, 0, th.conditional(alpha > 1, 1, alpha)))

                    else:
                        alphatest2.assign(th.Constant(1.0))
                        
                     # update sediment rate to ensure equilibrium at inflow
                    """
                    if cons_tracer:
                        sediment_rate.assign(depth.at([0,0])*ceq.at([0,0])/(coeff.at([0,0])))
                    else:
                        sediment_rate.assign(ceq.at([0,0])/(coeff.at([0,0])))                  
                    """
                    if t_new%(12*3600) >= 10800:
                        if t_new%(12*3600) < 32400:
                            sediment_rate.assign(-((10**-4)/2.65)*np.cos(2*np.pi*t_new/(12*3600)))
                    else:
                        sediment_rate.assign(0.0)                    
                    
                if bedload == True:
                    
                    # calculate angle of flow
                    calfa.interpolate(horizontal_velocity/th.sqrt(unorm))
                    salfa.interpolate(vertical_velocity/th.sqrt(unorm))
                    div_function.interpolate(th.as_vector((calfa, salfa)))
                        
                    if slope_eff == True:    
                        # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
                        # we use z_n1 and equals so that we can use an implicit method in Exner
                        slopecoef = (1 + beta*(z_n1.dx(0)*calfa + z_n1.dx(1)*salfa))
                    else:
                        slopecoef = th.Constant(1.0)
                        
                    if angle_correction == True:
                        # slope effect angle correction due to gravity
                        tt1.interpolate(th.conditional(1000*0.5*qfc*unorm > 10**(-10), th.sqrt(cparam/(1000*0.5*qfc*unorm)), th.sqrt(cparam/(10**(-10)))))
                        # add on a factor of the bed gradient to the normal
                        aa.assign(salfa + tt1*dzdy)
                        bb.assign(calfa + tt1*dzdx)
                        norm.assign(th.conditional(th.sqrt(aa**2 + bb**2) > 10**(-10), th.sqrt(aa**2 + bb**2),10**(-10)))
                        # we use z_n1 and equals so that we can use an implicit method in Exner
                        calfamod = (calfa + (tt1*z_n1.dx(0)))/norm
                        salfamod = (salfa + (tt1*z_n1.dx(1)))/norm  

                    if seccurrent == True:
                        # accounts for helical flow effect in a curver channel
                        
                        # again use z_n1 and equals so can use an implicit method in Exner
                        free_surface_dx = depth.dx(0) - z_n1.dx(0)
                        free_surface_dy = depth.dx(1) - z_n1.dx(1)

                        velocity_slide = (horizontal_velocity*free_surface_dy)-(vertical_velocity*free_surface_dx)
                        
                        tandelta_factor.interpolate(7*9.81*1000*depth*qfc/(2*alpha_secc*((horizontal_velocity**2) + (vertical_velocity**2))))

                        if angle_correction == True:
                            # if angle has already been corrected we must alter the corrected angle to obtain the corrected secondary current angle
                            t_1 = (TOB*slopecoef*calfamod) + (vertical_velocity*tandelta_factor*velocity_slide)
                            t_2 = (TOB*slopecoef*salfamod) - (horizontal_velocity*tandelta_factor*velocity_slide)
                        else:    
                            t_1 = (TOB*slopecoef*calfa) + (vertical_velocity*tandelta_factor*velocity_slide)
                            t_2 = ((TOB*slopecoef*salfa) - (horizontal_velocity*tandelta_factor*velocity_slide))

                        # calculated to normalise the new angles
                        t4 = th.sqrt((t_1**2) + (t_2**2))

                        # updated magnitude correction and angle corrections
                        slopecoef = t4/TOB

                        calfanew = t_1/t4
                        salfanew = t_2/t4
                 
                    
                    if bed_form == 'meyer':
                        # implement meyer-peter-muller bedload transport formula
                        thetaprime.interpolate(mu*(1000*0.5*qfc*unorm)/((rhos-1000)*9.81*average_size))

                        # if velocity above a certain critical value then transport occurs
                        phi.assign(th.conditional(thetaprime < thetacr, 0, 8*(thetaprime-thetacr)**1.5))
                        
                        # bedload transport flux with magnitude correction
                        qb_total = slopecoef*phi*th.sqrt(g*(rhos/1000 - 1)*average_size**3)
                    elif bed_form == 'soulsby':
                        abb.interpolate(th.conditional(depth >= average_size, 0.005*depth*((average_size/depth)**1.2)/coeff_soulsby, 0.005*depth/coeff_soulsby))
                        ucr_bed.interpolate(th.conditional(depth > d90, 0.19*(average_size**0.1)*(th.ln(4*depth/d90))/(th.ln(10)), 0.19*(average_size**0.1)*(th.ln(4))/(th.ln(10))))
                        s0_bed.interpolate(th.conditional((th.sqrt(unorm)-ucr_bed)**2.4 > 0, (th.sqrt(unorm)-ucr_bed)**2.4,0))
                        qb_total = slopecoef*abb*s0_bed*th.sqrt(unorm)
                    else:
                        print('Unrecognised bedload transport formula. Please choose "meyer" or "soulsby"')
                    # add time derivative to exner equation with a morphological scale factor    
                    f += (((1-porosity)*(z_n1 - z_n)/(dt*morfac)) * v)*fire.dx 
                    
                    # formulate bedload transport flux with correct angle depending on corrections implemented
                    if angle_correction == True and seccurrent == False:
                        qbx = qb_total*calfamod
                        qby = qb_total*salfamod
                    elif seccurrent == True:
                        qbx = qb_total*calfanew
                        qby = qb_total*salfanew                        
                    else:
                        qbx = qb_total*calfa
                        qby = qb_total*salfa                       
                    
                    # add bedload transport to exner equation
                    f += -(v*((qbx*n[0]) + (qby*n[1])))*fire.ds(1) -(v*((qbx*n[0]) + (qby*n[1])))*fire.ds(2) + (qbx*(v.dx(0)) + qby*(v.dx(1)))*fire.dx

                else:
                    # if no bedload transport component initialise exner equation with time derivative
                    f = (((1-porosity)*(z_n1 - z_n)/(dt*morfac)) * v)*fire.dx 
                    
                if suspendedload == True:
                    # add suspended sediment transport to exner equation multiplied by depth as the exner equation is not depth-averaged
                    f += - (qbsourcedepth*v)*fire.dx
                
                if update_bedlevel == True:
                    # solve exner equation using finite element methods
                    fire.solve(f == 0, z_n1)
                
                    # update bed
                    bathymetry_2d.assign(z_n1)

                if round(t_new, 2)%t_export == 0:
                    # calculate difference between original bathymetry and new bathymetry
                    diff_bathy.interpolate(-bathymetry_2d + orig_bathymetry)
                    bathy_file.write(bathymetry_2d)
                    diff_bathy_file.write(diff_bathy)             

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st


    # final time of simulation
    t_end = final_time/morfac
    print(final_time)
    print(t_end)
    print(morfac)
    # export interval in seconds
    t_export = np.round(t_end/72)
    
    th.print_output('Exporting to '+outputdir)
    
    x, y = th.SpatialCoordinate(mesh2d)
    
    # define function spaces
    P1_2d = th.FunctionSpace(mesh2d, 'DG', 1)
    vectorP1_2d = th.VectorFunctionSpace(mesh2d, 'DG', 1)
    V = th.FunctionSpace(mesh2d, 'CG', 1)
    vector_cg = th.VectorFunctionSpace(mesh2d, 'CG', 1)

    # define test functions on mesh
    v = fire.TestFunction(V)
    n = th.FacetNormal(mesh2d)
    z_n1 = fire.Function(V, name ="z^{n+1}")
    z_n = fire.Function(V, name="z^{n}")


    # define original bathymetry before bedlevel changes
    orig_bathymetry = th.Function(V).interpolate(bathymetry_2d)

    # calculate bed evolution
    diff_bathy = th.Function(V).interpolate(-bathymetry_2d + orig_bathymetry)


    # define output file for bed evolution
    bathy_file = th.File(outputdir + "/bathy_mc.pvd")
    #bathy_file.write(orig_bathymetry)
    diff_bathy_file = th.File(outputdir + "/diff_bathy.pvd")
    diff_bathy_file.write(diff_bathy)

    # define parameters
    g = th.Constant(9.81)
    porosity = th.Constant(0.4)


    ksp = th.Constant(3*average_size)
    a = th.Constant(ks/2)
    viscosity = th.Constant(10**(-6))

    # magnitude slope effect parameter
    beta = th.Constant(beta_fn)
    # angle correction slope effect parameters
    surbeta2 = th.Constant(surbeta2_fn)
    cparam = th.Constant((rhos-1000)*9.81*average_size*(surbeta2**2))
    # secondary current parameter
    alpha_secc = th.Constant(alpha_secc_fn)
    # maximum gradient allowed by sediment slide mechanism
    tanphi = math.tan(angle_fn*math.pi/180)
    # approximate mesh step size for sediment slide mechanism
    L = th.Constant(mesh_step_size)

    # calculate critical shields parameter thetacr
    R = th.Constant(rhos/1000 - 1)
    dstar = th.Constant(average_size*((g*R)/(viscosity**2))**(1/3))
    if max(dstar.dat.data[:] < 1):
        print('ERROR: dstar value less than 1')
    elif max(dstar.dat.data[:] < 4):
        thetacr = th.Constant(0.24*(dstar**(-1)))
    elif max(dstar.dat.data[:] < 10):
        thetacr = th.Constant(0.14*(dstar**(-0.64)))
    elif max(dstar.dat.data[:] < 20):
        thetacr = th.Constant(0.04*(dstar**(-0.1)))
    elif max(dstar.dat.data[:] < 150):
        thetacr = th.Constant(0.013*(dstar**(0.29)))        
    else:
        thetacr = th.Constant(0.055)

    # critical bed shear stress
    taucr = th.Constant((rhos-1000)*g*average_size*thetacr)   

    # calculate settling velocity
    if average_size <= 100*(10**(-6)):
            settling_velocity = th.Constant(9.81*(average_size**2)*((rhos/1000)-1)/(18*viscosity))
    elif average_size <= 1000*(10**(-6)):
            settling_velocity = th.Constant((10*viscosity/average_size)*(th.sqrt(1 + 0.01*((((rhos/1000) - 1)*9.81*(average_size**3))/(viscosity**2)))-1))
    else:
            settling_velocity = th.Constant(1.1*th.sqrt(9.81*average_size*((rhos/1000) - 1)))        
    
    # initialise velocity, elevation and depth
    elev_init, uv_init = initialise_fields(mesh2d, input_dir, outputdir)

    uv_cg = th.Function(vector_cg).interpolate(uv_init)

    elev_cg = th.Function(V).interpolate(elev_init)

    if wetting_and_drying:
        H = th.Function(V).project(elev_cg + bathymetry_2d)
        depth = th.Function(V).project(H + (0.5 * (th.sqrt(H ** 2 + wetting_alpha ** 2) - H)))
    else:
        depth = th.Function(V).project(elev_cg + bathymetry_2d)        

    old_bathymetry_2d = th.Function(V).interpolate(bathymetry_2d)

    horizontal_velocity = th.Function(V).interpolate(uv_cg[0])
    vertical_velocity = th.Function(V).interpolate(uv_cg[1])

    # define bed friction
    hc = th.Function(P1_2d).interpolate(th.conditional(depth > 0.001, depth, 0.001))
    aux = th.Function(P1_2d).interpolate(th.conditional(11.036*hc/ks > 1.001, 11.036*hc/ks, 1.001))
    qfc = th.Function(P1_2d).interpolate(2/(th.ln(aux)/0.4)**2)
    # skin friction coefficient
    hclip = th.Function(P1_2d).interpolate(th.conditional(ksp > depth, ksp, depth))
    cfactor = th.Function(P1_2d).interpolate(th.conditional(depth > ksp, 2*((2.5*th.ln(11.036*hclip/ksp))**(-2)), th.Constant(0.0)))
    # mu - ratio between skin friction and normal friction
    mu = th.Function(P1_2d).interpolate(th.conditional(qfc > 0, cfactor/qfc, 0))

    # calculate bed shear stress
    unorm = th.Function(P1_2d).interpolate((horizontal_velocity**2) + (vertical_velocity**2))
    TOB = th.Function(V).interpolate(1000*0.5*qfc*unorm)

    # define bed gradient
    dzdx = th.Function(V).interpolate(old_bathymetry_2d.dx(0))
    dzdy = th.Function(V).interpolate(old_bathymetry_2d.dx(1))

    if suspendedload == True:
        # deposition flux - calculating coefficient to account for stronger conc at bed
        B = th.Function(P1_2d).interpolate(th.conditional(a > depth, a/a, a/depth))
        ustar = th.Function(P1_2d).interpolate(th.sqrt(0.5*qfc*unorm))
        exp1 = th.Function(P1_2d).interpolate(th.conditional((th.conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), th.conditional((settling_velocity/(0.4*ustar)) -1 > 3, 3, (settling_velocity/(0.4*ustar))-1), 0))
        coefftest = th.Function(P1_2d).interpolate(th.conditional((th.conditional((settling_velocity/(0.4*ustar)) - 1 > 0, (settling_velocity/(0.4*ustar)) -1, -(settling_velocity/(0.4*ustar)) + 1)) > 10**(-4), B*(1-B**exp1)/exp1, -B*th.ln(B)))
        coeff = th.Function(P1_2d).interpolate(th.conditional(th.conditional(coefftest>10**(-12), 1/coefftest, 10**12)>1, th.conditional(coefftest>10**(-12), 1/coefftest, 10**12), 1))
        hloc = th.Function(V).interpolate(th.conditional(depth>0.01, depth, 0.01))
        
        if sus_form == 'vanrijn':
            # erosion flux - above critical velocity bed is eroded
            s0 = th.Function(P1_2d).interpolate((th.conditional(1000*0.5*qfc*unorm*mu > 0, 1000*0.5*qfc*unorm*mu, 0) - taucr)/taucr)
            ceq = th.Function(P1_2d).interpolate(0.015*(average_size/a) * ((th.conditional(s0 < 0, 0, s0))**(1.5))/(dstar**0.3))
        elif sus_form == 'soulsby':
            if d90 == 0:
                # if the value of d90 is unspecified set d90 = d50
                d90 = th.Constant(average_size)
            else:
                d90 = th.Constant(d90)
            coeff_soulsby = th.Constant((R*g*average_size)**1.2)
            ass = th.Constant(0.012*average_size*(dstar**(-0.6))/coeff_soulsby)
            ucr = th.Function(P1_2d).interpolate(0.19*(average_size**0.1)*(th.ln(4*hloc/d90))/(th.ln(10)))
            s0 = th.Function(P1_2d).interpolate(th.conditional((th.sqrt(unorm)-ucr)**2.4 > 0, (th.sqrt(unorm)-ucr)**2.4,0.0))
            ceq = th.Function(P1_2d).interpolate(ass*s0/depth)
        else:
            print('Unrecognised suspended sediment transport formula. Please choose "vanrijn" or "soulsby"')
   
        if convectivevel == True:
            # correction factor to advection velocity in sediment concentration equation

            Bconv = th.Function(P1_2d).interpolate(th.conditional(depth > 1.1*ksp, ksp/depth, ksp/(1.1*ksp)))
            Aconv = th.Function(P1_2d).interpolate(th.conditional(depth > 1.1* a, a/depth, a/(1.1*a)))
                    
            # take max of value calculated either by ksp or depth
            Amax = th.Function(P1_2d).interpolate(th.conditional(Aconv > Bconv, Aconv, Bconv))

            r1conv = th.Function(P1_2d).interpolate(1 - (1/0.4)*th.conditional(settling_velocity/ustar < 1, settling_velocity/ustar, 1))

            Ione = th.Function(P1_2d).interpolate(th.conditional(r1conv > 10**(-8), (1 - Amax**r1conv)/r1conv, th.conditional(r1conv < - 10**(-8), (1 - Amax**r1conv)/r1conv, th.ln(Amax))))

            Itwo = th.Function(P1_2d).interpolate(th.conditional(r1conv > 10**(-8), -(Ione + (th.ln(Amax)*(Amax**r1conv)))/r1conv, th.conditional(r1conv < - 10**(-8), -(Ione + (th.ln(Amax)*(Amax**r1conv)))/r1conv, -0.5*th.ln(Amax)**2)))

            alpha = th.Function(P1_2d).interpolate(-(Itwo - (th.ln(Amax) - th.ln(30))*Ione)/(Ione * ((th.ln(Amax) - th.ln(30)) + 1)))

            # final correction factor
            alphatest2 = th.Function(P1_2d).interpolate(th.conditional(th.conditional(alpha > 1, 1, alpha) < 0, 0, th.conditional(alpha > 1, 1, alpha)))
        else:
            alphatest2 = th.Function(P1_2d).interpolate(th.Constant(1.0))
            
        # update sediment rate to ensure equilibrium at inflow
        if cons_tracer:
            sediment_rate = th.Constant(0.0)#depth.at([0,0])*ceq.at([0,0])/(coeff.at([0,0])))
            testtracer = th.Function(P1_2d).interpolate(depth*ceq/coeff)
        else:
            sediment_rate = th.Constant(0.0)#ceq.at([0,0])/(coeff.at([0,0])))
            testtracer = th.Function(P1_2d).interpolate(ceq/coeff)   

        # get individual terms
        depo = th.Function(P1_2d).interpolate(settling_velocity*coeff)
        ero = th.Function(P1_2d).interpolate(settling_velocity*ceq)        
        
        # calculate depth-averaged source term for sediment concentration equation
        if cons_tracer:
            if depth_integrated:
                depth_int_sink = th.Function(P1_2d).interpolate(depo/depth)
                depth_int_source = th.Function(P1_2d).interpolate(ero)
            else:
                sink = th.Function(P1_2d).interpolate(depo/(depth**2))
                source = th.Function(P1_2d).interpolate(ero/depth)
            if tracer_init == None:
                qbsourcedepth = th.Function(P1_2d).interpolate(-(depo*sediment_rate/depth)+ ero)
            else:
                qbsourcedepth = th.Function(P1_2d).interpolate(-(depo*tracer_init)+ ero)        
        else:
            sink = th.Function(P1_2d).interpolate(depo/depth)
            source = th.Function(P1_2d).interpolate(ero/depth)        
            if tracer_init == None:
                qbsourcedepth = th.Function(P1_2d).interpolate(-(depo*sediment_rate)+ ero)
            else:
                qbsourcedepth = th.Function(P1_2d).interpolate(-(depo*tracer_init)+ ero)               
                    
    if bedload == True:
        #calculate angle of flow
        calfa = th.Function(V).interpolate(horizontal_velocity/th.sqrt(unorm))
        salfa = th.Function(V).interpolate(vertical_velocity/th.sqrt(unorm))
        div_function = th.Function(vector_cg).interpolate(th.as_vector((calfa, salfa)))
    
        if slope_eff == True:    
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            slopecoef = th.Function(V).interpolate(1 + beta*(dzdx*calfa + dzdy*salfa))
        else:
            slopecoef = th.Function(V).interpolate(th.Constant(1.0))
                        
        if angle_correction == True:
            # slope effect angle correction due to gravity
            tt1 = th.Function(V).interpolate(th.conditional(1000*0.5*qfc*unorm > 10**(-10), th.sqrt(cparam/(1000*0.5*qfc*unorm)), th.sqrt(cparam/(10**(-10)))))
            # add on a factor of the bed gradient to the normal
            aa = th.Function(V).interpolate(salfa + tt1*dzdy)
            bb = th.Function(V).interpolate(calfa + tt1*dzdx)
            norm = th.Function(V).interpolate(th.conditional(th.sqrt(aa**2 + bb**2) > 10**(-10), th.sqrt(aa**2 + bb**2),10**(-10)))
            calfamod = th.Function(V).interpolate(bb/norm)
            salfamod = th.Function(V).interpolate(aa/norm)
    
        if seccurrent == True:
            # accounts for helical flow effect in a curver channel
            free_surface_dx = th.Function(V).interpolate(elev_cg.dx(0))
            free_surface_dy = th.Function(V).interpolate(elev_cg.dx(1))
        
            velocity_slide = (horizontal_velocity*free_surface_dy)-(vertical_velocity*free_surface_dx)
                        
            tandelta_factor = th.Function(V).interpolate(7*9.81*1000*depth*qfc/(2*alpha_secc*((horizontal_velocity**2) + (vertical_velocity**2))))

            t_1 = (TOB*slopecoef*calfa) + (vertical_velocity*tandelta_factor*velocity_slide)
            t_2 = ((TOB*slopecoef*salfa) - (horizontal_velocity*tandelta_factor*velocity_slide))

            # calculated to normalise the new angles
            t4 = th.sqrt((t_1**2) + (t_2**2))

            # updated magnitude correction and angle corrections
            slopecoef = t4/TOB

                
            calfanew = t_1/t4
            salfanew = t_2/t4

        if bed_form == 'meyer':
            # implement meyer-peter-muller bedload transport formula
            thetaprime = th.Function(V).interpolate(mu*(1000*0.5*qfc*unorm)/((rhos-1000)*9.81*average_size))

            # if velocity above a certain critical value then transport occurs
            phi = th.Function(V).interpolate(th.conditional(thetaprime < thetacr, 0, 8*(thetaprime-thetacr)**1.5))

        elif bed_form == 'soulsby':
            if d90 == 0:
                d90 = th.Constant(average_size)
            coeff_soulsby = th.Constant((R*g*average_size)**1.2)
            abb = th.Function(P1_2d).interpolate(th.conditional(depth >= average_size, 0.005*depth*((average_size/depth)**1.2)/coeff_soulsby, 0.005*depth/coeff_soulsby))
            ucr_bed = th.Function(P1_2d).interpolate(th.conditional(depth > d90, 0.19*(average_size**0.1)*(th.ln(4*depth/d90))/(th.ln(10)), 0.19*(average_size**0.1)*(th.ln(4))/(th.ln(10))))
            s0_bed = th.Function(P1_2d).interpolate(th.conditional((th.sqrt(unorm)-ucr_bed)**2.4 > 0, (th.sqrt(unorm)-ucr_bed)**2.4,0))
        else:
            print('Unrecognised bedload transport formula. Please choose "meyer" or "soulsby"')
    # set up solver 
    solver_obj = th.solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    if suspendedload == True:
        # switch on tracer calculation if using sediment transport component
        options.solve_tracer = True
        options.use_tracer_conservative_form = cons_tracer
        options.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
        options.tracer_advective_velocity_factor = alphatest2
        if depth_integrated:
            options.tracer_depth_integ_source = depth_int_source
            options.tracer_depth_integ_sink = depth_int_sink
        else:
            options.tracer_source_2d = source
            options.tracer_sink_2d = sink
        options.check_tracer_conservation = True
    else:
        options.solve_tracer = False
        options.fields_to_export = ['uv_2d', 'elev_2d']
    options.use_lax_friedrichs_tracer = False
    # set bed friction
    if friction == 'nikuradse':
        options.quadratic_drag_coefficient = cfactor
    elif friction == 'manning':
        if friction_coef == 0:
            friction_coef = 0.02
        options.manning_drag_coefficient = th.Constant(friction_coef)
    else:
        print('Undefined friction')
    # set horizontal diffusivity parameter
    options.horizontal_diffusivity = th.Constant(diffusivity)
    options.horizontal_viscosity = th.Function(P1_2d).interpolate(sponge_viscosity*th.Constant(viscosity_hydro))
    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.use_wetting_and_drying = wetting_and_drying
    options.wetting_and_drying_alpha = th.Constant(wetting_alpha)
    options.norm_smoother = th.Constant(wetting_alpha)

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt
        
    #if options.solve_tracer:
    #    c = call.TracerTotalMassConservation2DCallback('tracer_2d',
    #                                     solver_obj, export_to_hdf5=True, append_to_log=False)
    #    solver_obj.add_callback(c, eval_interval='timestep')        
     
    # set boundary conditions
    swe_bnd, left_bnd_id, right_bnd_id, in_constant, out_constant, left_string, right_string = boundary_conditions_fn(bathymetry_2d, flag = 'morpho')
    
    for j in range(len(in_constant)):
        exec('constant_in' + str(j) + ' = th.Constant(' + str(in_constant[j]) + ')', globals())
    

    str1 = '{'
    if len(left_string) > 0:
        for i in range(len(left_string)):
            str1 += "'"+ str(left_string[i]) + "': constant_in" + str(i) + ","
        str1 = str1[0:len(str1)-1] + "}"
        exec('swe_bnd[left_bnd_id] = ' + str1)

    for k in range(len(out_constant)):
        exec('constant_out' + str(k) + '= th.Constant(' + str(out_constant[k]) + ')', globals())

    str2 = '{'
    if len(right_string) > 0:
        for i in range(len(right_string)):
            str2 += "'"+ str(right_string[i]) + "': constant_out" + str(i) + ","
        str2 = str2[0:len(str2)-1] + "}"
        exec('swe_bnd[right_bnd_id] = ' + str2)
         
    solver_obj.bnd_functions['shallow_water'] = swe_bnd
    
    print(solver_obj.bnd_functions['shallow_water'])
    if suspendedload == True:
        """
        solver_obj.bnd_functions['tracer'] = {left_bnd_id: {'value': sediment_rate}}
        
        for i in solver_obj.bnd_functions['tracer'].keys():
            if i in solver_obj.bnd_functions['shallow_water'].keys():
                solver_obj.bnd_functions['tracer'][i].update(solver_obj.bnd_functions['shallow_water'][i])
        for i in solver_obj.bnd_functions['shallow_water'].keys():
            if i not in solver_obj.bnd_functions['tracer'].keys():
                solver_obj.bnd_functions['tracer'].update({i:solver_obj.bnd_functions['shallow_water'][i]})        
        """
        print('switched off tracer init')
        # set initial conditions
        solver_obj.bnd_functions['tracer'] = swe_bnd
        if tracer_init == None:       
            solver_obj.assign_initial_conditions(uv=uv_init, elev= elev_init, tracer = testtracer)
        else:
            solver_obj.assign_initial_conditions(elev= elev_init, uv = uv_init, tracer = tracer_init)            
        
    else:
        # set initial conditions
        solver_obj.assign_initial_conditions(uv=uv_init, elev= elev_init)
        


    return solver_obj, update_forcings_tracer, diff_bathy, diff_bathy_file, outputdir

def export_final_state(inputdir, uv, elev,):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    th.print_output("Exporting fields for subsequent simulation")
    chk = th.DumbCheckpoint(inputdir + "/velocity", mode=th.FILE_CREATE)
    chk.store(uv, name="velocity")
    th.File(inputdir + '/velocityout.pvd').write(uv)
    chk.close()
    chk = th.DumbCheckpoint(inputdir + "/elevation", mode=th.FILE_CREATE)
    chk.store(elev, name="elevation")
    th.File(inputdir + '/elevationout.pvd').write(elev)
    chk.close()
    
    plex = elev.function_space().mesh()._plex
    viewer = PETSc.Viewer().createHDF5(inputdir + '/myplex.h5', 'w')
    viewer(plex)
    
    
def initialise_fields(mesh2d, inputdir, outputdir,):
    """
    Initialise simulation with results from a previous simulation
    """
    DG_2d = th.FunctionSpace(mesh2d, 'DG', 1)
    # elevation
    with th.timed_stage('initialising elevation'):
        chk = th.DumbCheckpoint(inputdir + "/elevation", mode=th.FILE_READ)
        elev_init = th.Function(DG_2d, name="elevation")
        chk.load(elev_init)
        th.File(outputdir + "/elevation_imported.pvd").write(elev_init)
        chk.close()
    # velocity
    with th.timed_stage('initialising velocity'):
        chk = th.DumbCheckpoint(inputdir + "/velocity" , mode=th.FILE_READ)
        V = th.VectorFunctionSpace(mesh2d, 'DG', 1)
        uv_init = th.Function(V, name="velocity")
        chk.load(uv_init)
        th.File(outputdir + "/velocity_imported.pvd").write(uv_init)
        chk.close()
        return  elev_init, uv_init,
