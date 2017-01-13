import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as bread
from prospect.sources import CSPBasis
import os
import sys


filename = sys.argv[1]
if os.path.exists('run_params.py'):
	os.system('rm run_params.py')
	os.system('cp ../../code/new_params.py .')
else:
	os.system('cp ../../code/new_params.py .')
	
res, _, mod = bread.results_from(filename) 
sps = CSPBasis(**res['run_params'])

def walkers(res=res):

    pfig = bread.param_evol(res)
    plt.show()
    return

def make_sed(typee, mod=mod, res=res, sps=sps):
    wl_arr = np.logspace(3,6,1000)
    obs = res['obs']
    wave = [f.wave_effective for f in obs['filters']]
    theta = mod.theta
    obs['wavelength'] = wave
    sed, photometry, x = mod.mean_model(theta, obs,sps)
    wave2 = np.array(wave) / 1E10
    nu_wave = 299792458. / np.array(wave2)
    nuFnu = sed * nu_wave
    nPhot = nu_wave * photometry
    if typee=='Fnu':
        return wave, sed, photometry
    elif typee=='nuFnu':
        return wave, nuFnu, nPhot

def make_nicer_sed(typee, mod=mod, res=res, sps=sps):
    wl_arr = np.logspace(3,7,10000)
    obs = res['obs']
    theta = mod.theta
    obs['wavelength'] = wl_arr
    sed, photometry, x = mod.mean_model(theta, obs,sps)
    wl_arr2 = wl_arr / 1E10
    nu_wave = 299792458./wl_arr2
    nuFnu = nu_wave*sed
    if typee=='Fnu':
        return wl_arr, sed, photometry
    elif typee=='nuFnu':
        return wl_arr, nuFnu, photometry
def load_raw_sed(typee, res=res):
    obs = res['obs']
    wave = [f.wave_effective for f in obs['filters']]
    wave2 = np.array(wave) / 1E10
    wave_nu = 299792458. / wave2
    maggies = obs['maggies']
    nuFnu = wave_nu * maggies
    unc = obs['maggies_unc']
    unc_nuFnu = wave_nu * unc
    if typee=="nuFnu":
        return wave, nuFnu, unc_nuFnu
    elif typee=="Fnu":
        return wave, maggies, unc
def plot_sed(typee,num,res=res):
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.ion()
    wl, fl, phot = make_sed(typee)
    wll,fll, phot2 = make_nicer_sed(typee)
    owl,ofl,ounc = load_raw_sed(typee)
    fig1 = plt.figure(num)
    frame1=fig1.add_axes((.1,.3,.8,.6))
    
    plt.errorbar(owl,ofl,yerr=ounc,fmt='ks',label='Composite SED')
    plt.plot(wll,fll,label='Best Fit Spectrum [model]')
    plt.plot(wl,phot,'r^',label='Best Fit Photometry [model]')
    #frame1.axes.xaxis.set_ticklabels([])
    frame1.xaxis.set_ticks_position("top")
    #frame1.axes.get_xaxis().set_visible(False)
    #frame1.xaxis.set_major_formatter(plt.NullFormatter())
    plt.grid()
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc=4)
    obs = res['obs']
    title = obs['objname']
    #plt.title(title, y=1.08)
    if typee=='Fnu':
        plt.ylabel(r'$F_{\nu} (maggies)$',fontsize=16)
    elif typee=='nuFnu':
        plt.ylabel(r'$\nu F_{\nu}$',fontsize=16)
    resid = ofl-phot
    resid = resid / ounc
    frame2=fig1.add_axes((.1,.1,.8,.2))  
    plt.plot(wl, resid, 'k.')
    plt.axhline(0,linestyle='--')
    plt.xscale('log')
    plt.xlabel(r'log ($\lambda_{rest}$) [$\AA$]',fontsize=16)
    plt.grid()
    frame2.axes.set_yticks(frame2.axes.get_yticks()[:-1])
    plt.ylabel(r'$\chi$',fontsize=16)
    plt.show()
    return



#Main Script
plot_sed('Fnu',1)
plot_sed('nuFnu',2)
walkers()

