import numpy as np
from prospect.models import priors, sedmodel
from prospect.sources import CSPBasis
tophat = priors.tophat
from sedpy.observate import load_filters
import os
# --------------
# RUN_PARAMS
# --------------

run_params = {'verbose':True,
              'debug':False,
              'outfile':'../results/iteration10/TRY__',                    #### SET OUTPUT DIR/FILE
              # Fitter parameters
              'nwalkers':128,                                              #### SET nwalkers
              'nburn':[128, 128, 128], 'niter':512,                        #### SET nburn
              'do_powell': False,
              'ftol':0.5e-5, 'maxfev':5000,
              'initial_disp':0.1,
              # Obs data parameter
      	      'phottable':'../../SEDs/v2/seds/iteration10_sed.txt',        #### SET INPUT SED (CHANGE NUM)
              'objname':'iteration10',                                     #### SET OBJNAME
              'filt_dir':'../../SEDs/v2/composite_filters/iteration_10/'   #### SET INPUT FILTER DIRECTORY
              'logify_spectrum':False,
              'normalize_spectrum':False,
              'wlo':3750, 'whi':7200,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# OBS
# --------------

def load_obs(objname='objname', phottable='phottable',filt_dir='filt_dir', **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.
    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.
    :param phottable:
        Name (and path) of the ascii file containing the photometry.
    :returns obs:
        Dictionary of observational data.
    """
    catalog = np.loadtxt(phottable)
    catalog = np.transpose(catalog)

    fluxes = catalog[0] #Should already be in Maggies
    fl_maggies = fluxes 
    fl_err = catalog[1] #Should Already Be in Maggies
    er_maggies = fl_err 

    ####################################
    # Create list of filename strings
    filter_files = os.listdir(filt_dir) #only filter files (in increasing num) should be present in the filter directory
    ####################################


    # Build output dictionary. 
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.

    obs['filters'] = load_filters(filter_files, directory=filt_dir)

    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.

    obs['maggies'] = fl_maggies
    obs['maggies_unc'] = er_maggies
    obs['maggies_unc'] = (.1*np.max(fl_maggies))*er_maggies  #### THIS THING INVOLVES A LOT OF TOYING WITH
 
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = [True]*len(fl_maggies)
    # We have no spectrum
    obs['wavelength'] = None

    # Add unessential bonus info.  This will be sored in output
    #obs['dmod'] = catalog[ind]['dmod']
    obs['objname'] = objname
    return obs


# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=0, compute_vega_mags=False, **extras):
    sps = CSPBasis(zcontinuous=zcontinuous,
                   compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Gaussian Process
# ------------------

def load_gp(**extras):
    return None, None

# --------------
# MODEL_PARAMS
# --------------

# You'll note below that we have 5 free parameters:
# mass, logzsol, tage, tau, dust2
# Each has tophat priors. They are all scalars.
#
# The other parameters are all fixed, but we want to explicitly set their
# values.

model_params = []

# --- Distance ---
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':4.0}})

# --- SFH --------
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'type'
                    })

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': True,
                        'init': 10,
                        'init_disp': 5,
                        'units': r'M_\odot',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.01, 'maxi':1000}})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.1,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior_function': tophat,
                        'prior_args': {'mini':-1, 'maxi':2.0}})

#model_params.append({'name': 'pmetals', 'N': 1,
#                        'isfree': False,
#                        'init': -99.0,
#                        'init_disp': 0.1,
#                        'units': r'$\log (Z/Z_\odot)$',
#                        'prior_function': tophat,
#                        'prior_args': {'mini':-99.0, 'maxi':2.0}})
                        
model_params.append({'name': 'tau', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 10,
                        'units': 'Gyr',
                        'prior_function':priors.logarithmic,
                        'prior_args': {'mini':0.1, 'maxi':100}})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 5.0,
                        'init_disp': 3.0,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.101, 'maxi':14.0}})

model_params.append({'name': 'sfstart', 'N': 1,
                        'isfree':False,
                        'init': 0.01,
                        'init_disp': 0.0,
                        'units': 'Gyr',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.01, 'maxi':14.0}})

model_params.append({'name': 'tburst', 'N': 1,
                        'isfree': True,
                        'init': 2.0,
                        'init_disp': 1.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':13.0}})

model_params.append({'name': 'fburst', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.9}})

# --- Dust ---------
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': True,
                        'init': 0.35,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':2.0}})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.35,
                        'reinit': True,
                        'init_disp': 0.3,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':2.0}})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': -0.7,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-1.5, 'maxi':2.0}})

#model_params.append({'name': 'dust1_index', 'N': 1,
#                        'isfree': False,
#                        'init': -1.0,
#                        'units': '',
#                        'prior_function':tophat,
#                        'prior_args': {'mini':-1.5, 'maxi':-0.5}})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': tophat,
                        'prior_args':{'mini':3.0, 'maxi':9.0} })

model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'index'})

model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 1.0,
                        'prior_function': tophat,
                        'prior_args': {'mini':0.1,'maxi':25.0},
                        'units': 'MMP83 local MW intensity'})
model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 3.5,
                        'init_disp': 1.0,
                        'prior_function': tophat,
                        'prior_args': {'mini':.1,'maxi':10.0},
                        'units': 'MMP83 local MW intensity'})
model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': .1,
                        'prior_function': logarithmic,
                        'prior_args': {'mini':0.0,'maxi':0.3},
                        'units': 'MMP83 local MW intensity'})

# --- Stellar Pops ------------
model_params.append({'name': 'tpagb_norm_type', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': 'index'})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'agb_dust', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': 'index'})

# --- Nebular Emission ------

# Here is a really simple function that takes a **dict argument, picks out the
# `logzsol` key, and returns the value.  This way, we can have gas_logz find
# the value of logzsol and use it, if we uncomment the 'depends_on' line in the
# `gas_logz` parameter definition.
#
# One can use this kind of thing to transform parameters as well (like making
# them linear instead of log, or divide everything by 10, or whatever.) You can
# have one parameter depend on several others (or vice versa).  Just remember
# that a parameter with `depends_on` must always be fixed.

def stellar_logzsol(logzsol=0.0, **extras):
    return logzsol

model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'index'})

model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'depends_on': stellar_logzsol,
                        'prior_function':tophat,
                        'prior_args': {'mini':-2.0, 'maxi':0.5}})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': True,
                        'init': -2.0,
                        'units': '',
                        'prior_function':tophat,
                        'prior_args': {'mini':-4, 'maxi':-1}})

#--- IGM Absorbtion -------
model_params.append({'name': 'add_igm_absorbtion', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'units': 'index'})


# --- Calibration ---------
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': 'mags',
                        'prior_function':tophat,
                        'prior_args': {'mini':0.0, 'maxi':0.2}})

def load_model(**extras):
    # In principle (and we've done it) you could have the model depend on
    # command line arguments (or anything in run_params) by making changes to
    # `model_params` here before instantiation the SedModel object.  Up to you.
    return sedmodel.SedModel(model_params)

