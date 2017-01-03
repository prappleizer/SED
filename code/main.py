#################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from pysynphot import observation
from pysynphot import spectrum
import os
#################################################################
fname = '../input_files/cosmos-1.deblend.herschel.v1.0.cat'
conversion_file = '../input_files/ID_conversion_v2.txt'
bins_file = '../input_files/Bins_v4.7.dat'
z_file = '../input_files/cosmos-1.bc03.v4.7.fout'
filt_dir = '../input_files/transmission_curves/'
filters = [filt_dir+'PacsFilter_blue.txt',filt_dir+'PacsFilter_green.txt',filt_dir+'sp250.dat',filt_dir+'sp350.dat']
#################################################################
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
	IN: iteration number (int), path to herschel catalog (str), path to v4.7-->v5.1 file (str), path to bins file, path to redshift file
	OUT: composite dict containing wavelengths, fluxes, errors, and v4.7/v5.1 IDs for a given iteration
	'''
	def __init__(self, iteration_num, catalog_path,conversion_file,bins_file,z_file):
		self.composite = {} #the final container
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
		#loop through all v5.1 IDs and find them in the herschel catalog, noting their index for slicing in a moment
		#at the same time, loop through v4.7 IDs and find them in the redshift-catalog and note their index for slicing in a moment
		# scaling and iteration information comes from bins- so we check where old_ids (the ids in bins) is the current scanning v4ID.
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
		
		# arrange fluxes, errors, wavelengths, and IDS into composite dictionary.
		self.composite['fluxes'] = fluxes
		self.composite['errors'] = errors
		self.composite['wavelengths'] = wls
		self.composite['v4IDs'] = v4ID
		self.composite['v5IDs'] = v5ID
		self.composite['zout'] = zout_arr

	def load_dyas(self,iteration,err_out=False):
		'''
		Load optical SED stacks made by Dyas Utomo
		IN: iteration number
		OUT: wl (A), flux (maggies)
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
		flux = data[1] #Jansky
		flux_maggies = flux / 3631. #maggies
		error = data[2]
		error_maggies = error / 3631.
		if self.err_out == False:
			return wl, flux_maggies
		elif self.err_out ==True:
			return wl, flux_maggies, error_maggies

	def plot_sed(self):
		'''
		Plot optical SED with herschel data for same galaxies (pre stack)
		'''
		plt.plot(self.composite['wavelengths']['24'],self.composite['fluxes']['F24'],'.',label=r'$24\mu m$')
		plt.plot(self.composite['wavelengths']['100'],self.composite['fluxes']['F100'],'.',label=r'$100\mu m$')
		plt.plot(self.composite['wavelengths']['160'],self.composite['fluxes']['F160'],'.',label=r'$160\mu m$')
		plt.plot(self.composite['wavelengths']['250'],self.composite['fluxes']['F250'],'.',label=r'$250\mu m$')
		plt.plot(self.composite['wavelengths']['350'],self.composite['fluxes']['F350'],'.',label=r'$350\mu m$')
		wl,fl = self.load_dyas(self.iteration_num)
		plt.plot(wl,fl,'ks',label='Optical SED')
		plt.xscale('log')
		plt.yscale('log')
		#plt.legend(loc=2)
		#plt.show()
		


		

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
	all_ir_fluxes = np.concatenate((herschel_obj.composite['fluxes']['F100'], herschel_obj.composite['fluxes']['F160'],herschel_obj.composite['fluxes']['F250'],herschel_obj.composite['fluxes']['F350']))
	all_ir_wavelengths = np.concatenate((herschel_obj.composite['wavelengths']['100'], herschel_obj.composite['wavelengths']['160'],herschel_obj.composite['wavelengths']['250'],herschel_obj.composite['wavelengths']['350']))
	# just zout 4 times in a row (since the order of the fluxes/wl is consistent filter to filter)
	redshifts = np.concatenate((herschel_obj.composite['zout'],herschel_obj.composite['zout'],herschel_obj.composite['zout'],herschel_obj.composite['zout']))
	# tag everything from four filters with a 1,2,3 or 4 (to create composite filters later).
	which_filter = np.concatenate((np.ones(len(herschel_obj.composite['fluxes']['F100'])), np.ones(len(herschel_obj.composite['fluxes']['F100']))*2, np.ones(len(herschel_obj.composite['fluxes']['F100']))*3, np.ones(len(herschel_obj.composite['fluxes']['F100']))*4))
	
	# Clever bit of rearranging- above we have all the ir fluxes from the different filters one after another in whatever order they were in
	# We also have the wl, redshifts, and which filter they are in the same order. 
	# This step reorders everything so that the wavelengths are in increasing order, but all matching sets of wl, fl, z, filt stay the same wrt each other
	# That is, reorder every row of the array based on a rearrangement of one of them

	composite = np.array([all_ir_wavelengths,all_ir_fluxes,redshifts,which_filter])
	composite = np.transpose(composite)
	composite = composite[composite[:,0].argsort()]
	composite = np.transpose(composite)

    #Pull everything back out of composite (if this is slow can try to rework)
	wls = composite[0]
	fls = composite[1]
	zs = composite[2]
	filt = composite[3]

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
	stacked_wl = np.insert(stacked_wl,0,np.median(herschel_obj.composite['wavelengths']['24']))
	median_fl = np.insert(median_fl,0,np.median(herschel_obj.composite['fluxes']['F24']))
	# Error calculation for stacks:
	# mips separately due to legacy shit
	ir_errors = stack_error(fl_bins)
	mips_error = stack_error([herschel_obj.composite['fluxes']['F24']])
	error_arr = np.concatenate((mips_error,ir_errors))

	return stacked_wl, median_fl, error_arr, z_bins,filt_bins, herschel_obj
	
def plot_it(iteration_num, num_stacks):
	wl, fl, err, herschel_obj = create_stacks(iteration_num,num_stacks) 
	herschel_obj.plot_sed()
	plt.plot(wl,fl,'ks',ms=7,label='IR stacks (me)')
	plt.legend(loc=2)
	plt.show()


class SED(object):

	def __init__(self,iteration_num,num_stacks):
		self.iteration_num = iteration_num
		self.num_stacks = num_stacks
		self.ir_wl, self.ir_fl, self.ir_err, self.z_bins, self.filt_bins, self.herschel_obj = create_stacks(self.iteration_num,self.num_stacks)
		self.d_wl, self.d_fl, self.derr = self.herschel_obj.load_dyas(self.iteration_num,err_out=True)
		self.d_wl = self.d_wl[:-1] # drop dyas's mips point in favor of my own (can change)
		self.d_fl = self.d_fl[:-1] # drop dyas's mips point in favor of my own (can change)
		self.derr = self.derr[:-1] # drop dyas's mips point in favor of my own (can change)

		self.wavelengths = np.concatenate((self.d_wl,self.ir_wl))
		self.fluxes = np.concatenate((self.d_fl,self.ir_fl))
		self.errors = np.concatenate((self.derr,self.ir_err))
		self.out_array = np.column_stack((self.wavelengths.flatten(),self.fluxes.flatten(),self.errors.flatten())) 

	def create_filters(self):
		# Load the filters 
		pacs_blue = np.loadtxt(filters[0],skiprows=2,dtype=[('wl',float),('transmission',float)])
		pacs_blue['wl']*=1000.
		pacs_green = np.loadtxt('/Users/ipasha/Documents/research/pacs/PacsFilters/PacsFilter_green.txt',skiprows=2,dtype=[('wl',float),('transmission',float)])
		pacs_green['wl']*=1000.
		spire250 = np.loadtxt('/Users/ipasha/Documents/research/pacs/PacsFilters/sp250.dat',dtype=[('wl',float),('transmission',float)])
		spire350 = np.loadtxt('/Users/ipasha/Documents/research/pacs/PacsFilters/sp350.dat',dtype=[('wl',float),('transmission',float)])
		mips_filt = np.loadtxt('/Users/ipasha/Documents/research/pacs/PacsFilters/mips_24um.dat',dtype=[('wl',float),('transmission',float)])
		def rebin_spec(wave, specin, wavnew):
			spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
			f = np.ones(len(wave))
			filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
			obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
			return obs.binflux
		def calculations(z,filt):
			common_wl=np.arange(10000,10000000,1000)
			drs_wl = filt['wl'] / (1+z)
			#normalized_tr = filt['transmission'] / simps(filt['transmission'],drs_wl) #faster to run simps just once on final thing right?
			rebinned_spec = rebin_spec(drs_wl,filt['transmission'],common_wl)
			return rebinned_spec

		self.common_wl_arr = np.arange(10000,10000000,1000) # extended wl array to map all the filter files onto (can adjust resolution)
		
		#perform calculation for bin 0 (mips) -separate because it is not contiguous with the FIR points.
		mips_comp_filt = np.zeros(len(self.common_wl_arr))
		for z in self.herschel_obj.composite['zout']:
			mips_comp_filt += calculations(z,mips_filt)
		mips_comp_filt /= simps(mips_comp_filt,self.common_wl_arr)

		#perform calculation for herschel bins (IR stacks) - this works for any number of stacks you choose to create
		stack_filters = [mips_comp_filt]
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
			bin_comp_filter /= simps(bin_comp_filter,self.common_wl_arr) #normalize area to 1
			#print 'composite filter area: ', simps(bin_comp_filter,self.common_wl_arr) #confirm normalization to 1
			stack_filters.append(bin_comp_filter)
		return stack_filters

	def plot_filters(self):
		self.comp_filts = self.create_filters()
		plt.plot(self.common_wl_arr, self.comp_filts[0])
		plt.plot(self.common_wl_arr, self.comp_filts[1])
		plt.plot(self.common_wl_arr, self.comp_filts[2])
		plt.plot(self.common_wl_arr, self.comp_filts[3])	
		plt.xscale('log')
		plt.show()	
	def file_out(self,fname):
		self.fname=fname
		np.savetxt(self.fname,self.out_array)
	def plot(self,display='Fnu'):
		self.display=display
		if self.display=='Fnu':
			plt.errorbar(self.wavelengths,self.fluxes,yerr=self.errors,fmt='s',ms=7)
			plt.ylabel(r'$F_{\nu}$')
		elif self.display=='nuFnu':
			nu = 299792458. / (self.wavelengths*1E-10) 
			nuFnu = nu*self.fluxes
			nuFnu_err=nu*self.errors
			plt.errorbar(self.wavelengths,nuFnu,yerr=nuFnu_err,fmt='s',ms=7)
			plt.ylabel(r'$\nu F_{\nu}$')

		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel(r'$\lambda$ [$\AA$]')
		plt.show()



def main(iteration_num, num_stacks):
	sed = SED(iteration_num,num_stacks)
	#outname='../SEDs/seds/iteration' + str(iteration_num) + '_sed.txt'
	#sed.file_out(outname)
	filt_stacks = sed.create_filters()
	outdir = '../SEDs/composite_filters/iteration_' + str(iteration_num)
	last_fname = os.listdir(outdir)[-1]
	last_num = int(last_fname[7:11]) #extract the number of the last file in the directory 
	wl_arr = np.arange(10000,10000000,1000)
	print len(filt_stacks)
	for i in range(len(filt_stacks)):
		filt_out = np.column_stack((wl_arr,filt_stacks[i]))
		outnum = last_num+1+i
		outname = outdir + '/filter_' + str(outnum) + '.par'
		np.savetxt(outname,filt_out)



for i in range(1,33):
	print 'Working on SED # ', i
	main(i,3)
















