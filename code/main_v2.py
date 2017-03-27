#################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from pysynphot import observation
from pysynphot import spectrum
import os
import sys
#################################################################
####################### Filepaths ###############################
fname = '../input_files/cosmos-1.deblend.herschel.v1.0.cat'
conversion_file = '../input_files/ID_conversion_v2.txt'
bins_file = '../input_files/Bins_v4.7.dat'
z_file = '../input_files/cosmos-1.bc03.v4.7.fout'
filt_dir = '../input_files/IR_transmission_curves/'
filters = [filt_dir+'mips_24um.dat',filt_dir+'PacsFilter_blue.txt',filt_dir+'PacsFilter_green.txt',filt_dir+'sp250.dat',filt_dir+'sp350.dat']

output_version_number = '3'
#################################################################
################### Auxilliary Functions ########################
def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

def stack_error(fl_bins):
	""" Return the bootstap resampled error on a stack of data points (median)
	Parameters
	-----------
	fl_bins: array containing subarrays of fluxes (len number of bins)
	
	Results
	-----------
	returns comp_errors (len number of bins)
	"""
	comp_errors = []
	for i in fl_bins:
	    arr = []
	    for j in range(10000):
	        boot = bootstrap_resample(i) 
	        arr.append(np.median(boot))
	    xy = np.histogram(arr, bins=78)
	    x = xy[1]
	    x = x[0:-1]
	    mean = np.median(x)
	    diffs = (x-mean)**2
	    summ = sum(diffs)
	    summ = summ/(len(diffs)-1)
	    final_err = np.sqrt(summ)
	    comp_errors.append(final_err)
	return comp_errors

class Herschel(object):
	'''
	Constructs a Herschel object containing the relevant info on an SED from all the various sources. 
	Constructs for a single iteration number. 
	Parameters
	----------
	iteration_num (int): the iteration number of the categorized SED being constructed (from Kriek+ 2011)
	catalog_path (str): filepath to the hershcel deblended photometry catalog [fname]
	conversion_file (str): filepath to the conversion file containing the v4.7-->v5.1 corresponding ID numbers for sources in the catalogs
	bins_file (str): filepath to the "bins" file containing the iteration for every galaxy (in the v4.7 catalog)
	z_file (str): filepath to the "fout" file containing the redshifts for all v4.7 sources
	
	Results
	---------
	returns Herschel Object 
	'''
	def __init__(self, iteration_num, catalog_path,conversion_file,bins_file,z_file):
		self.iteration_num = iteration_num
		###############################################################################
		dtypes = [  ('ID',float),('RA',float),('DEC',float),('F24',float),\
		('E24',float),('F100',float),('E100',float),('F160',float),('E160',float),\
		('F250',float),('E250',float),('F350',float),('E350',float),('F500',float),('E500',float),\
		('flag_match',float),('flxratio',float),('idmulti1',float), ('idmulti2',float),\
		('idmulti3',float),('idmulti4',float),('idmulti5',float),('flag_spire500',float)  ]
		alldata = np.loadtxt(catalog_path,dtype=dtypes) #load the full herschel catalog
		print '25 percent complete...'
		################################################################################
		usecols=(0,3,5)
		dtypes = [ ('ID',float),('iteration',float),('scale',float) ]
		try:
			bin_info = np.loadtxt(bins_file,usecols=usecols,dtype=dtypes) #load the file containing the iteration information for each galaxy (v4.7)
			old_ids = [f['ID'] for f in bin_info]
			iteration = [f['iteration'] for f in bin_info]
			scale = [f['scale'] for f in bin_info]
		except:
			'Error loading bin info file...'
		###############################################################################
		usecols=(0,1)
		dtypes = [('ID',float),('z',float)]
		try:
			z_fout = np.loadtxt(z_file,usecols=usecols,dtype=dtypes) #load file containing redshift information for each galaxy
			zIDs = np.array([f['ID'] for f in z_fout])
			zs = np.array([f['z'] for f in z_fout])
		except:
			print 'Error loading redshift file...'
		###############################################################################
		dtypes=[ ('v4ID',float),('v5ID',float)  ]
		conversion_array = np.loadtxt(conversion_file,dtype=dtypes) #load file containing v4.7-->v5.1 translation
		v4ID = conversion_array['v4ID']
		v5ID = conversion_array['v5ID']
		###############################################################################
		#initialize empty container lists
		herschel_ind = []
		zout_ind = []
		scales = []
		iterations = []
		###############################################################################
		#loop through all v5.1 IDs and find them in the herschel catalog, noting their index for slicing in a moment
		#at the same time, loop through v4.7 IDs and find them in the redshift-catalog and note their index for slicing in a moment
		# scaling and iteration information comes from bins- so we check where old_ids (the ids in bins) is the current scanning v4ID.
		###############################################################################
		for i in range(len(v5ID)):
			cat_indx = np.where(alldata['ID'] == v5ID[i])[0]
			herschel_ind.append(cat_indx)
			zout_indx = np.where(z_fout['ID']==v4ID[i])[0]
			zout_ind.append(zout_indx)
			try:
				indx = int(np.where(old_ids == v4ID[i])[0])
				iterations.append(iteration[indx])
				scales.append(scale[indx])
			except:
				print 'old ID list (from Bins_v4.7) does not contain current ID in v4ID: %s' %(v4ID[i])
				continue
		###############################################################################		
		scales = np.array(scales)
		iterations = np.array(iterations)
		self.sources = [alldata[f] for f in herschel_ind] #restrict to those v5IDs we are using in this project
		flux_indices = np.where(iterations==iteration_num)[0] #find the indices in iterations which correspond to the iteration being instantiated in this object. iterations has already been limited to the galaxies in this sample.
		scales = np.array([scales[f] for f in flux_indices])
		v4ID = np.array([v4ID[f] for f in flux_indices])
		v5ID = np.array([v5ID[f] for f in flux_indices])
		zout_arr = np.concatenate([[zs[f] for f in zout_ind][i] for i in flux_indices])
		print '50 percent complete...'
		cwl={'24':240000.,'100':1026174.64,'160':1671355.25,'250':2556456.6,'350':3585285.2} #define central wavelengths of mips/herschel filters
		# initialize wavelength dictionary with value for each filter
		# wavelength formula cwl/(1+z)
		wls = {}
		wls['24'] = np.ones(len(zout_arr))*cwl['24'] / (1+zout_arr)
		wls['100'] = np.ones(len(zout_arr))*cwl['100'] / (1+zout_arr)
		wls['160'] = np.ones(len(zout_arr))*cwl['160'] / (1+zout_arr)
		wls['250'] = np.ones(len(zout_arr))*cwl['250'] / (1+zout_arr)
		wls['350'] = np.ones(len(zout_arr))*cwl['350'] / (1+zout_arr)
		
		# initialize flux dictionary with value for each filter
		# factor of 0.27541 applied to all fluxes to make conversion mJy-->maggies (*1000 (Jy), then /3631 (maggies))
		# scales is an array corresponding (hopefully) to scaling of each flux needed- array multiply to do it. 
		# factor of 1/(z)**2 for some godawful reason I don't remember but it should be right. 
		fluxes = {}
		fluxes['F24'] = [np.array([f[0]['F24']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		fluxes['F100'] = [np.array([f[0]['F100']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		fluxes['F160'] = [np.array([f[0]['F160']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		fluxes['F250'] = [np.array([f[0]['F250']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		fluxes['F350'] = [np.array([f[0]['F350']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		
		print '75 percent complete...'
		
		# initialize error dictionary with value for each filter
		# same multiplicative factors as fluxes. 
		errors = {}
		errors['E24'] = [np.array([f[0]['E24']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		errors['E100'] = [np.array([f[0]['E100']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		errors['E160'] = [np.array([f[0]['E160']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		errors['E250'] = [np.array([f[0]['E250']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		errors['E350'] = [np.array([f[0]['E350']*0.27541 for f in self.sources])[i] for i in flux_indices]*scales / zout_arr**2
		
		self.fluxes = fluxes
		self.errors = errors
		self.wavelengths = wls
		self.v4ID = v4ID
		self.v5ID = v5ID
		self.zout = zout_arr

	def load_dyas(self,iteration,err_out=False):
		'''
		Load optical SED stacks made by Dyas Utomo
		IN: iteration number
		OUT: wl (A), flux (maggies), flux error (maggies)
		'''
		self.err_out=err_out
		#it-to-type because dyas names files by sed number not iteration number 
		#it_to_type = {1:3,2:2, 3:25, 4:21, 5:49, 6:16, 7:28, 8:9 , 9:30, 10:13, 11:31, 12:11, 13:18, 14:1, 15:15, 16:7, 17:17, 18:27, 19:14, 20:23,21:32,22:5, 23:22, 24:26, 25:10,26:20,27:12,28:29,29:19,30:8, 31:24, 32:6} 
		it_to_type = {1:3,2:2, 3:25, 4:21, 5:4, 6:16, 7:28, 8:9, 9:30, 10:13, 11:31, 12:11, 13:18, 14:1, 15:15, 16:7, 17:17, 18:27, 19:14, 20:23,21:32,22:5, 23:22, 24:26, 25:10,26:20,27:12,28:29,29:19,30:8, 31:24, 32:6} 
		sedtype = it_to_type[iteration]
		pref = '/Users/ipasha/RESEARCH/SEDs/dyas_seds/type'
		mid = str(sedtype)
		final = '.cat'
		filen = pref + mid+ final
		data = np.genfromtxt(filen, usecols=(0,3,4))
		data = np.transpose(data)
		wl = data[0] #micron
		wl = wl*10000 #Angstrom
		flux = data[1] #Jansky (even though the files say uJy in them)
		flux_maggies = flux / 3631. #maggies
		error = data[2]
		error_maggies = error / 3631.
		if self.err_out == False:
			return wl, flux_maggies
		elif self.err_out ==True:
			return wl, flux_maggies, error_maggies

	def plot_sed(self,incl_optical=True,nufnu=False):
		'''
		Plot optical SED with herschel data for same galaxies (pre stack)
		'''
		if nufnu==False:
			plt.plot(self.wavelengths['24'],self.fluxes['F24'],'.',label=r'$24\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['100'],self.fluxes['F100'],'.',label=r'$100\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['160'],self.fluxes['F160'],'.',label=r'$160\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['250'],self.fluxes['F250'],'.',label=r'$250\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['350'],self.fluxes['F350'],'.',label=r'$350\mu m$',alpha=.2,color='k')
			fl_list = ['F24','F100','F160','F250','F350']
			wl_list = ['24','100','160','250','350']
			for i in range(len(fl_list)):
				neg_vals = np.where(self.fluxes[fl_list[i]]<0)[0]
				vals = (np.ones(len(neg_vals))*min(j for j in self.fluxes[fl_list[3]] if j>0)) / 2.
				plt.plot(self.wavelengths[wl_list[i]][neg_vals],vals,'kv',ms=3)
		elif nufnu==True:
			#nu = 299792458. / (self.wavelengths*1E-10)
			#mips_nuFnu = 
			concat_wl = np.concatenate((self.wavelengths['24'],self.wavelengths['100'],self.wavelengths['160'],self.wavelengths['250'],self.wavelengths['350']))
			concat_vfl = np.concatenate((self.fluxes['F24']*(299792458./ (self.wavelengths['24']*1E-10)),self.fluxes['F100']*(299792458./ (self.wavelengths['100']*1E-10)),self.fluxes['F160']*(299792458./ (self.wavelengths['160']*1E-10)),self.fluxes['F250']*(299792458./ (self.wavelengths['250']*1E-10)),self.fluxes['F350']*(299792458./ (self.wavelengths['350']*1E-10))))
			plt.plot(self.wavelengths['24'],self.fluxes['F24']*(299792458./ (self.wavelengths['24']*1E-10)),'.',label=r'$24\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['100'],self.fluxes['F100']*(299792458./ (self.wavelengths['100']*1E-10)),'.',label=r'$100\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['160'],self.fluxes['F160']*(299792458./ (self.wavelengths['160']*1E-10)),'.',label=r'$160\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['250'],self.fluxes['F250']*(299792458./ (self.wavelengths['250']*1E-10)),'.',label=r'$250\mu m$',alpha=.2,color='k')
			plt.plot(self.wavelengths['350'],self.fluxes['F350']*(299792458./ (self.wavelengths['350']*1E-10)),'.',label=r'$350\mu m$',alpha=.2,color='k')
			fl_list = ['F24','F160','F100','F160','F250','F350']
			wl_list = ['24','160','100','160','250','350']
			neg_vals = np.where(concat_vfl<0)[0]
			vals = np.ones(len(neg_vals))*min(concat_vfl[np.where(concat_vfl>0)[0]]) / 2.
			plt.plot(concat_wl[neg_vals],vals,'kv',ms=3)
		if incl_optical:
			wl,fl = self.load_dyas(self.iteration_num)
			plt.plot(wl,fl,'ks',label='Optical SED')
		plt.xscale('log')
		plt.yscale('log')
		#plt.legend(loc=2)
		#plt.show()
#################################################################		
#################################################################
#################################################################

def create_stacks(iteration_num,num_stacks):
	'''
	Takes herschel data for galaxy sample and creates n median stacks 
	IN: iteration number (int): num_stacks (int) [not including mips, which is automatically treated as one stack]
	'''
	stack_wls = {}
	stack_fluxes = {}

	herschel_obj = Herschel(iteration_num,fname,conversion_file,bins_file,z_file)
	print 'Herschel Object Created...'
	#create concatenated IR flux arrays (not including mips, handled separately)
	all_ir_fluxes = np.concatenate((herschel_obj.fluxes['F100'], herschel_obj.fluxes['F160'],herschel_obj.fluxes['F250'],herschel_obj.fluxes['F350']))
	all_ir_wavelengths = np.concatenate((herschel_obj.wavelengths['100'], herschel_obj.wavelengths['160'],herschel_obj.wavelengths['250'],herschel_obj.wavelengths['350']))
	# just zout 4 times in a row (since the order of the fluxes/wl is consistent filter to filter)
	redshifts = np.concatenate((herschel_obj.zout,herschel_obj.zout,herschel_obj.zout,herschel_obj.zout))
	# tag everything from four filters with a 1,2,3 or 4 (to create composite filters later).
	which_filter = np.concatenate((np.ones(len(herschel_obj.fluxes['F100'])), np.ones(len(herschel_obj.fluxes['F160']))*2, np.ones(len(herschel_obj.fluxes['F250']))*3, np.ones(len(herschel_obj.fluxes['F350']))*4))
	
	# Clever bit of rearranging- above we have all the ir fluxes from the different filters one after another in whatever order they were in
	# We also have the wl, redshifts, and which filter they are in the same order. 
	# This step reorders everything so that the wavelengths are in increasing order, but all matching sets of wl, fl, z, filt stay the same wrt each other
	# That is, reorder every row of the array based on a rearrangement of one of them

	full_sed_data = np.array([all_ir_wavelengths,all_ir_fluxes,redshifts,which_filter])
	full_sed_data = np.transpose(full_sed_data)
	full_sed_data = full_sed_data[full_sed_data[:,0].argsort()]
	full_sed_data = np.transpose(full_sed_data)
	#The above step should have ordered the arrays by lowest to highest wavelength, and then carried along all the other info.
    #Pull everything back out of full_sed_data (maybe reworks this)
	wls = full_sed_data[0]
	fls = full_sed_data[1]
	zs = full_sed_data[2]
	filt = full_sed_data[3]

	int_division = int(len(wls) / num_stacks) #figure out the max amount num_stacks goes into something len wls without remainder
	end_index = num_stacks*int_division #this has to be the end index for np.split, which requires an even division be possible
	shortened_wl = np.array(wls[0:end_index])
	shortened_fl = np.array(fls[0:end_index])  # shorten the arrays to only go up to end_index
	shortened_z = np.array(zs[0:end_index])
	shortened_which_filt = np.array(filt[0:end_index])
	wl_bins = np.split(shortened_wl,num_stacks)
	fl_bins = np.split(shortened_fl,num_stacks) #split them into n subarrays
	z_bins = np.split(shortened_z, num_stacks)
	filt_bins = np.split(shortened_which_filt, num_stacks)
	
	wl_bins[-1] = np.append(wl_bins[-1], wls[end_index:]) #toss the leftovers into the last bin (sketchy)
	fl_bins[-1] = np.append(fl_bins[-1], fls[end_index:])
	z_bins[-1] = np.append(z_bins[-1], zs[end_index:-1])
	filt_bins[-1] = np.append(filt_bins[-1], filt[end_index:-1])

	stacked_wl = []  
	median_fl = []
	    
	for i in wl_bins:
	    wl_out = np.median(i)
	    stacked_wl.append(wl_out)
	for i in fl_bins:
	    fl_out = np.median(i)
	    median_fl.append(fl_out)
	stacked_wl = np.array(stacked_wl)
	median_fl = np.array(median_fl)
	stacked_wl = np.insert(stacked_wl,0,np.median(herschel_obj.wavelengths['24']))
	median_fl = np.insert(median_fl,0,np.median(herschel_obj.fluxes['F24']))
	# Error calculation for stacks:
	ir_errors = stack_error(fl_bins) #call stack error function from top of file
	mips_error = stack_error([herschel_obj.fluxes['F24']])
	error_arr = np.concatenate((mips_error,ir_errors))

	return stacked_wl, median_fl, error_arr, z_bins,filt_bins, herschel_obj
	
def plot_it(iteration_num, num_stacks):
	wl, fl, err, herschel_obj = create_stacks(iteration_num,num_stacks) 
	herschel_obj.plot_sed()
	plt.plot(wl,fl,'ks',ms=7,label='IR stacks (me)')
	plt.legend(loc=2)
	plt.show()

#################################################################
#################################################################
#################################################################

class SED(object):
	""" Create an SED object containing the full arrays of wavelengths, fluxes, and errors for the stacks.
	    Contains methods for plotting as well as for creating the composite filters for newly created IR stacks. 

	   	Parameters
	   	-----------
	   	iteration_num: The desired iteration number of the composite SED desired (from Kriek+ 2011)
	   	num_stacks: The desired numbers of evenly spaced stacks in the FIR data (one stack is always created separately for mips 24um) (3 is typical)

	   	Results
	   	-----------
	   	returns SED object. 

	"""

	def __init__(self,iteration_num,num_stacks):
		self.iteration_num = iteration_num
		self.num_stacks = num_stacks
		self.ir_wl, self.ir_fl, self.ir_err, self.z_bins, self.filt_bins, self.herschel_obj = create_stacks(self.iteration_num,self.num_stacks)
		self.d_wl, self.d_fl, self.derr = self.herschel_obj.load_dyas(self.iteration_num,err_out=True)
		self.d_mips_wl, self.d_mips_fl, self.d_mips_er = self.d_wl[-1], self.d_fl[-1], self.derr[-1]
		self.d_wl = self.d_wl[:-1] # drop dyas's mips point in favor of my own (can change)
		self.d_fl = self.d_fl[:-1] # drop dyas's mips point in favor of my own (can change)
		self.derr = self.derr[:-1] # drop dyas's mips point in favor of my own (can change)

		self.wavelengths = np.concatenate((self.d_wl,self.ir_wl))
		self.fluxes = np.concatenate((self.d_fl,self.ir_fl))
		self.errors = np.concatenate((self.derr,self.ir_err))
		self.out_array = np.column_stack((self.wavelengths.flatten(),self.fluxes.flatten(),self.errors.flatten())) 

	def create_filters(self):
		""" for the number of IR stacks created, synthesize a composite filter for the stack from the individual
			transmission curves used to observe the photometry included in each stack. 

			Parameters
			-----------
			None (inherent method)

			Results
			-----------
			stack_filters (array_like): array containing subarrays with the synthesized composite transmission curves (len num_stacks)

		"""
		# Load the filters 
		mips_filt = np.loadtxt(filters[0],dtype=[('wl',float),('transmission',float)])
		pacs_blue = np.loadtxt(filters[1],skiprows=2,dtype=[('wl',float),('transmission',float)]) #load filter files from internal dir
		pacs_blue['wl']*=10000.
		pacs_green = np.loadtxt(filters[2],skiprows=2,dtype=[('wl',float),('transmission',float)])
		pacs_green['wl']*=10000.
		spire250 = np.loadtxt(filters[3],dtype=[('wl',float),('transmission',float)])
		spire350 = np.loadtxt(filters[4],dtype=[('wl',float),('transmission',float)])
		def rebin_spec(wave, specin, wavnew):
			""" rebin a spectrum/transmission curve onto a new wavelength array resolution (uses external packages)

				Parameters
				-----------
				wave (array_like): array containing wavelengths for curve 
				specin (array_like): array containing values (must be len(wave)) 
				wavnew (array_like): wavelength array to map spectrum onto 
				-----------
				returns binflux (array_like): transmission curve/spectrum remapped/interpolated to lie on new wavelength array 
			"""
			spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
			f = np.ones(len(wave))
			filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
			obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
			return obs.binflux
		def calculations(z_arr,filt):
			common_wl=np.arange(10000,10000000,1000)
			drs_wl = filt['wl'] / (1+z_arr)
			#normalized_tr = filt['transmission'] / simps(filt['transmission'],drs_wl) #faster to run simps just once on final thing right?
			rebinned_spec = rebin_spec(drs_wl,filt['transmission'],common_wl)
			return rebinned_spec

		self.common_wl_arr = np.arange(10000,10000000,1000) # extended wl array to map all the filter files onto (can adjust resolution)
		
		#perform calculation for bin 0 (mips) -separate because it is not contiguous with the FIR points.
		mips_comp_filt = np.zeros(len(self.common_wl_arr))
		for z in self.herschel_obj.zout:
			mips_comp_filt += calculations(z,mips_filt)
		#plt.plot(self.common_wl_arr,mips_comp_filt,lw=2,color='b',label='mips')
		mips_comp_filt /= simps(mips_comp_filt,self.common_wl_arr)

		#perform calculation for herschel bins (IR stacks) - this works for any number of stacks you choose to create
		stack_filters = [mips_comp_filt]
		color=['r','g','k']
		for i in range(len(self.z_bins)):
			bin_comp_filter = np.zeros(len(self.common_wl_arr))
			for j in range(len(self.z_bins[i])):
				if int(self.filt_bins[i][j]) == 1:
					rebinned_spec = calculations(self.z_bins[i][j],pacs_blue)
				elif int(self.filt_bins[i][j]) == 2:
					rebinned_spec = calculations(self.z_bins[i][j],pacs_green)
				elif int(self.filt_bins[i][j]) == 3: 
					rebinned_spec = calculations(self.z_bins[i][j],spire250)
				elif int(self.filt_bins[i][j]) == 4:
					rebinned_spec = calculations(self.z_bins[i][j],spire350)
				bin_comp_filter += rebinned_spec
				#if i==1:
					#plt.plot(self.common_wl_arr,rebinned_spec)
			#label='comp for filt' + str(i)
			#plt.plot(self.common_wl_arr,bin_comp_filter,color=color[i],lw=2,label=label)
			bin_comp_filter /= simps(bin_comp_filter,self.common_wl_arr) #normalize area to 1

			#print 'composite filter area: ', simps(bin_comp_filter,self.common_wl_arr) #confirm normalization to 1
			stack_filters.append(bin_comp_filter)
		#for i in stack_filters:
		#	plt.plot(self.common_wl_arr,i)
		#plt.legend()
		#plt.show()
		return stack_filters

	def plot_filters(self):
		self.comp_filts = self.create_filters()
		plt.plot(self.common_wl_arr, self.comp_filts[0],label='Mips composite')
		plt.plot(self.common_wl_arr, self.comp_filts[1],label='stack one composite')
		plt.plot(self.common_wl_arr, self.comp_filts[2],label='stack two composite')
		plt.plot(self.common_wl_arr, self.comp_filts[3],label='stack three composite')	
		plt.xscale('log')
		plt.legend()
		plt.show()	
	def file_out(self,fname):
		self.fname=fname
		np.savetxt(self.fname,self.out_array)
	def plot(self,display='Fnu',incl_raw=False,save_fig=False):
		plt.figure()
		self.display=display
		if self.display=='Fnu':
			#plt.errorbar(self.wavelengths,self.fluxes,yerr=self.errors,fmt='s',color='k',ms=7)
			plt.ylabel(r'$F_{\nu}$')
			for i in range(len(self.fluxes)):
				if self.fluxes[i] < 0: 
					plt.plot(self.wavelengths[i],self.errors[i],'kv',ms=10)
				else:
					plt.errorbar(self.wavelengths[i],self.fluxes[i],yerr=self.errors[i],fmt='s',color='k',ms=7)
			if incl_raw:
				self.herschel_obj.plot_sed(incl_optical=False,nufnu=False)
		elif self.display=='nuFnu':
			nu = 299792458. / (self.wavelengths*1E-10) 
			nuFnu = nu*self.fluxes
			nuFnu_err=nu*self.errors
			#plt.errorbar(self.wavelengths,nuFnu,yerr=nuFnu_err,fmt='s',color='k',ms=7)
			plt.ylabel(r'$\nu F_{\nu}$')
			for i in range(len(self.fluxes)):
				if nuFnu[i] < 0: 
					plt.plot(self.wavelengths[i],nuFnu_err[i],'kv',ms=10)
				else:
					plt.errorbar(self.wavelengths[i],nuFnu[i],yerr=nuFnu_err[i],fmt='s',color='k',ms=7)
			if incl_raw:
				self.herschel_obj.plot_sed(incl_optical=False,nufnu=True)
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel(r'$\lambda$ [$\AA$]')
		plt.tight_layout()
		tit = 'SED for iteration: ' + str(self.iteration_num)
		plt.title(tit)
		if save_fig:
			title = '../plots/raw_composite_seds/iteration_' + str(self.iteration_num) + '.pdf'
			plt.savefig(title)
			plt.close()
		elif save_fig==False:
			plt.show()

		




def main(iteration_num, num_stacks,output_version_number=output_version_number):
	sed = SED(iteration_num,num_stacks)
	outname='../SEDs/v'+output_version_number+'/seds/iteration' + str(iteration_num) + '_sed.txt'
	sed.file_out(outname)
	filt_stacks = sed.create_filters()
	outdir = '../SEDs/v' +output_version_number+'/composite_filters/iteration_' + str(iteration_num)
	last_fname = os.listdir(outdir)[-1]
	last_num = int(last_fname[7:11]) #extract the number of the last file in the directory 
	wl_arr = np.arange(10000,10000000,1000)
	print len(filt_stacks)
	for i in range(len(filt_stacks)):
		filt_out = np.column_stack((wl_arr,filt_stacks[i]))
		outnum = last_num+1+i
		outname = outdir + '/filter_' + str(outnum) + '.par'
		np.savetxt(outname,filt_out)

def run_main(num_stacks):
	count = 0
	for i in range(1,33):
		count += 1
		print 'Working on SED: ', count
		main(i,num_stacks)

def save_raw_figs():
	for i in range(1,33):
		sed = SED(i,3)
		sed.plot('nuFnu',incl_raw=True,save_fig=True)




def compare_dyas():
	phot_fluxes = []
	dyas_fluxes = []
	phot_errors = []
	dyas_errors = []
	for i in range(1,33):
		print 'Working on SED: ', i 
		sed = SED(i,3)
		phot_fluxes.append(sed.ir_fl[0])
		phot_errors.append(sed.ir_err[0])
		dyas_fluxes.append(sed.d_mips_fl)
		dyas_errors.append(sed.d_mips_er)
	out_arr = np.column_stack((phot_fluxes,phot_errors,dyas_fluxes,dyas_errors))
	np.savetxt('mips_comparison_mean.txt',out_arr)

		#plt.errorbar(phot_mips_fl,d_mips_fl,yerr=d_mips_er,xerr=phot_mips_er,fmt='o',ms=15)
	#plt.show()
def actually_compare():
	a = np.loadtxt('mips_comparison_mean.txt')
	a = np.transpose(a)
	pf = a[0]
	pe = a[1]
	df = a[2]
	de = a[3]
	'''
	plt.errorbar(pf,df,yerr=de,xerr=pe,fmt='o',ms=20,alpha=0.2)
	offset = 1
	for i in range(len(pf)):
		plt.annotate(str(i+1),xy=(pf[i],df[i]),xytext=(-offset,-offset))
	x = np.linspace(0,1,10000)
	plt.plot(x,x,'k')
	for i in range(len(pf)):
		if pf[i] < 0:
			plt.plot(1E-4,df[i],'<',ms=15,color='b',alpha=0.2)
			plt.annotate(str(i+1),xy=[1.03E-4,df[i]])
	#plt.xscale('log')
	#plt.yscale('log')
	plt.xlabel(r'Herschel Photometry [$F_{\nu}$]')
	plt.ylabel(r'Image Stacking [$F_{\nu}$]')
	plt.show()
	'''
	fig, ax = plt.subplots()
	for i in range(len(pf)):
		if pf[i] > 0:
			ax.errorbar(pf[i],df[i],yerr=de[i],xerr=pe[i],fmt='o',ms=20,alpha=0.2,color='b')
	offset = 1.0 
	xx = np.linspace(0,1,10000)
	#ax.set_xlim(min(pf)-offset, max(pf)+ offset)
	#ax.set_ylim(min(df)-offset, max(df)+ offset)
	count = 0
	for x,y in zip(pf,df):
		count += 1
		text = str(count)
		fontsize, aspect_ratio = (12, 0.5) # needs to be adapted to font
		width = len(text) * aspect_ratio * fontsize 
		height = fontsize
		a = ax.annotate(text,  xy=(x,y), xytext=(-width/2.0,-height/2.0), textcoords='offset points')
	for i in range(len(pf)):
		if pf[i] < 0:
			error = np.array([[pe[i],0]]).T
			plt.errorbar(pe[i],df[i],yerr=de[i],xerr=error,fmt='<',ms=25,color='b',alpha=0.2)
			ontsize, aspect_ratio = (12, 0.5) # needs to be adapted to font
			width = len(text) * aspect_ratio * fontsize 
			height = fontsize
			plt.annotate(str(i+1),xy=[pe[i],df[i]],xytext=(width/2.9,-height/2.3), textcoords='offset points')
	ax.plot(xx,xx,'k')
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel(r'Herschel Photometry [maggies]')
	ax.set_ylabel(r'Image Stacking [maggies]')
	plt.tight_layout()
	plt.show()








