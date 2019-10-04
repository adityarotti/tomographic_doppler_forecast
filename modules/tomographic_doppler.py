##################################################################################################
# Author 1 : Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester
# Author 2 : Nidhi Joshi Pant, Department of Physics & Astronomy, University of the Western Cape
# Date created:  16 September 2019
# Date modified: 16 September 2019
##################################################################################################

import camb
import numpy as np
import collections
from camb import model
from itertools import product
import auto_doppler_cov as adc
import cross_doppler_cov as cdc
from scipy.integrate import quad
from scipy.interpolate import interp1d
from camb.sources import GaussianSourceWindow, SplinedSourceWindow


class ska_spectroscopic_doppler_forecast(object):

	def __init__(self,bnu=3.,dw=1.):
		'''Empty for now'''
		self.bnu=bnu
		self.dw=dw
	
		data=np.loadtxt("../data/fitting_data.txt")
		self.fitc=collections.OrderedDict()
		for i,uJy in enumerate(data[:,0]):
			self.fitc[uJy]=data[i,1:]
		self.flux_cutoff_options=self.fitc.keys()
	
	def calc_doppler_error(self,lmin=128,lmax=2048,nbin=10):
		self.ell_max=np.linspace(lmin,lmax,nbin,dtype=np.int)
		self.err=np.zeros(len(self.ell_max),dtype=np.float64)
		self.simple_err=np.zeros(len(self.ell_max),dtype=np.float64)
		e=np.ones(len(self.adr_d1d2),dtype=np.float64)
		for i,iell in enumerate(self.ell_max):
			self.evaluate_covariance_mat(lmax=iell)
			self.cov_mat_inv=np.linalg.inv(self.cov_mat)
			self.err[i]=1./np.dot(e,np.dot(self.cov_mat_inv,e))
			self.simple_err[i]=1./np.dot(e,np.dot(np.linalg.inv(np.diag(np.diag(self.cov_mat))),e))
	
	def evaluate_covariance_mat(self,lmax=[]):
		'''Populate the covariance matrix'''
		self.cov_mat=np.zeros((len(self.zpair),len(self.zpair)),dtype=np.float64)
		self.cov_diag=np.zeros(len(self.zpair),dtype=np.float64)
		
		if lmax==[]:
			lmax=self.lmax
		elif lmax>self.lmax:
				print "Error, lmax cannot be larger than", self.lmax
				print "Setting lmax to largest possible lmax"
				lmax=int(self.lmax)
	
		for i,zp1 in enumerate(self.zpair):
			for ip,zp2 in enumerate(self.zpair[i:]):
				j=ip+i
				self.return_cls_for_estimators(i,j)
				self.cov_mat[i,j]=cdc.cross_doppler_forecast(cl1=self.cl_est1[:,0:lmax+1],cl2=self.cl_est2[:,0:lmax+1],clc=self.cl_cross[:,0:lmax+1],bnu=self.bnu,dw=self.dw,lmax=int(lmax))/self.fsky
				self.cov_mat[j,i]=self.cov_mat[i,j]
				self.cov_diag[i]=adc.auto_doppler_forecast(cl=self.cl_est1,bnu=self.bnu,dw=self.dw,lmax=int(self.lmax))
	
					
	def return_cls_for_estimators(self,zp1,zp2):
		key=self.adr_d1d2[zp1] ; idx=key.find("x") ; w1=key[0:idx] ; w2=key[idx+1:]
		self.auto_cov_key1=[w1+"x"+w1,w2+"x"+w2,w1+"x"+w2]
		self.cl_est1=np.zeros((5,self.lmax+1),dtype=np.float64)
		if self.nzbin>1:
			self.cl_est1[0,]=self.cls[key]-(self.cls[self.adr_dg[zp1][0]]+self.cls[self.adr_dg[zp1][1]])/2. # Modulation Est1
		else:
			self.cl_est1[0,]=self.cls[key] # Modulation Est1
		self.cl_est1[1,]=self.cls[key]     # Aberration Est1
		for ip, ckey in enumerate(self.auto_cov_key1):
			self.cl_est1[ip+2,:]=self.cls[ckey] + self.nl[ckey]

	
		key=self.adr_d1d2[zp2] ; idx=key.find("x") ; w3=key[0:idx] ; w4=key[idx+1:]
		self.auto_cov_key2=[w3+"x"+w3,w4+"x"+w4,w3+"x"+w4]
		self.cl_est2=np.zeros((5,self.lmax+1),dtype=np.float64)
		if self.nzbin>1:
			self.cl_est2[0,]=self.cls[key]-(self.cls[self.adr_dg[zp2][0]]+self.cls[self.adr_dg[zp2][1]])/2. # Modulation Est2
		else:
			self.cl_est2[0,]=self.cls[key] 	# Modulation Est2
		self.cl_est2[1,]=self.cls[key] 		# Aberration Est2
		for ip, ckey in enumerate(self.auto_cov_key2):
			self.cl_est2[ip+2,:]=self.cls[ckey] + self.nl[ckey]


		self.cross_cov_key=[w1+"x"+w3,w2+"x"+w4,w1+"x"+w4,w2+"x"+w3]
		self.cl_cross=np.zeros((len(self.cross_cov_key),self.lmax+1),dtype=np.float64)
		for ip, ckey in enumerate(self.cross_cov_key):
			self.cl_cross[ip,:]=self.cls[ckey] + self.nl[ckey]
	

	def return_spectra(self,verbose=False):
		# Spectra needed for covariance
		SW=[]
		self.adr_d=collections.OrderedDict() ; self.adr_g=collections.OrderedDict()
		for i,zp in enumerate(self.z_centroid):
			# The ordering of the sline source window is critical to the analysis.
			# Its set such that odd windows correspond to window & even windows correspond to window derivatives.
			SW.append(SplinedSourceWindow(z=self.z,W=self.window[zp],bias=self.bias[zp],dlog10Ndm=self.magnification[zp]))
			self.adr_d[zp]="W"+str(2*i+1) # Which window for the the redshift slice.(ODD)
			SW.append(SplinedSourceWindow(z=self.z,W=self.dwindow[zp],bias=self.bias[zp],dlog10Ndm=self.magnification[zp]))
			self.adr_g[zp]="W"+str(2*i+2) # Which window for the derivative of the window function --> g-term (EVEN)
		self.pars.SourceWindows = SW
		self.results = camb.get_results(self.pars)
		temp_cls=self.results.get_source_cls_dict()
		
		# Constructing the dictionary of the required power spectra.
		self.zpair=([zp for zp in product(self.z_centroid,self.z_centroid)])
		self.zpair=list(sorted(({tuple(sorted(zp)) for zp in self.zpair})))
		
		self.adr_d1d2=collections.OrderedDict() ; self.adr_dg=collections.OrderedDict() ; self.adr_auto=collections.OrderedDict()
		for i,zp in enumerate(self.zpair):
			# Address for all the auto/cross power spectra --> d1xd2
			self.adr_d1d2[i]=self.adr_d[zp[0]] + "x" + self.adr_d[zp[1]]
			if zp[0]==zp[1]:
				self.adr_auto[i]=self.adr_d1d2[i]
			# Address for all the spectra needed to evaluate the dg-term --> d1xg2 + d2xg1
			self.adr_dg[i]=[self.adr_d[zp[0]] + "x" + self.adr_g[zp[1]],self.adr_d[zp[1]] + "x" + self.adr_g[zp[0]]]
	
		
		self.adr_all_spec=np.unique([adr for adr_pair in self.adr_dg.values() for adr in adr_pair] + list(self.adr_d1d2.values()))
#		self.adr_all_spec=self.adr_all_spec)
		self.cls=collections.OrderedDict() ; self.nl=collections.OrderedDict()
		for key in self.adr_all_spec:
			idx=key.find("x")
			keyp=key[idx+1:] + "x" + key[0:idx] #; print keyp,key
			for i,zp in enumerate(self.z_centroid):
				if self.adr_g[zp] in key:
					# Renormalizing the dg spectra
					self.cls[key]=temp_cls[key][:self.lmax+1]*self.norm_ell*self.dwindow_norm[zp]/self.window_norm[zp]
					self.cls[keyp]=self.cls[key]
					if verbose:
						print self.adr_g[zp],key,self.dwindow_norm[zp]/self.window_norm[zp]
		
			if key not in self.cls.keys():
				self.cls[key]=temp_cls[key][:self.lmax+1]*self.norm_ell
				self.cls[keyp]=self.cls[key]
				if verbose:
					print "Added the spectra with key", key, "to the cls array"
			
			
			# Populate the noise power spectrum
			if key in self.adr_auto.values():
				# The auto correlation noise power spectrum.
				# This needs to be updated with the mean number of galaxies per steradian in the particular redshift bin.
				idx=key.find("x") ; d_adr=key[0:idx] ; z_idx=(self.adr_d.values()).index(d_adr)
				self.nl[key]=np.ones_like(self.cls[key])/self.nbar[self.z_centroid[z_idx]]
				self.nl[keyp]=self.nl[key]
			else:
				# The cross correlation noise power spectrum, assumed to be zero.
				self.nl[key]=np.zeros_like(self.cls[key])
				self.nl[keyp]=self.nl[key]
				
	def setup_obs_param(self,flux_cutoff,zmin=0.1, zmax=2.9,z_step=1e-4,area_obs=30000.):
		self.flux_cut=flux_cutoff
		self.area_obs=area_obs # Square degrees
		self.total_sky_area=4.*np.pi*(180./np.pi)**2.
		self.fsky=self.area_obs/self.total_sky_area
		self.zmin=zmin ; self.zmax=zmax ; self.z_step=z_step
		self.z=np.arange(self.zmin,self.zmax+self.z_step,self.z_step)
		self.c1=self.fitc[self.flux_cut][0]
		self.c2=self.fitc[self.flux_cut][1]
		self.c3=self.fitc[self.flux_cut][2]
		self.c4=self.fitc[self.flux_cut][3]
		self.c5=self.fitc[self.flux_cut][4]
		self.dndz=self.return_dndz(self.z)*self.fsky
		self.total_gal_num=self.return_integral(self.z,self.dndz)
				
	def setup_window_functions(self, nzbin=5, z_olap=0.1, wtype="tanh", taper_width=0.01,normalize=False):
		self.nzbin=nzbin ; self.z_olap=z_olap ; self.taper_width=taper_width
		z_edge = np.linspace(self.zmin+5*self.taper_width, self.zmax-5*self.taper_width, self.nzbin+1) # Bin edges
		self.z_bin=zip(z_edge[:-1]-z_olap/2.,z_edge[1:]+z_olap/2.)
		self.dz=np.mean([zbin[1]-zbin[0] for zbin in self.z_bin])
		self.z_centroid=np.zeros(len(self.z_bin))
		
		self.bias=collections.OrderedDict()
		self.magnification=collections.OrderedDict()
		for i in range(len(self.z_bin)):
			self.z_centroid[i]=np.mean(self.z_bin[i])
			self.bias[self.z_centroid[i]]=self.return_bias(self.z_centroid[i])
			self.magnification[self.z_centroid[i]]=0.
	
		self.total_window=np.zeros_like(self.z)
		self.tom_window=collections.OrderedDict()
		self.window=collections.OrderedDict() ; self.dwindow=collections.OrderedDict()
		self.window_norm=collections.OrderedDict()
		self.dwindow_norm=collections.OrderedDict()
		self.nbar=collections.OrderedDict()
		for i,zp in enumerate(self.z_centroid):
			self.tom_window[zp]=self.return_tanh_window(self.z,self.z_bin[i][0],self.z_bin[i][1],taper_width=self.taper_width)
			self.tom_window[zp]=self.tom_window[zp]/self.return_integral(self.z,self.tom_window[zp])
			self.window[zp]=self.tom_window[zp]*self.dndz ; self.nbar[zp]=self.return_integral(self.z,self.window[zp])
			self.dwindow[zp]=self.return_derivative(self.z,self.window[zp]*(1.+self.z),eps=self.z_step)
			self.window_norm[zp]=self.return_integral(self.z,self.window[zp])
			self.dwindow_norm[zp]=self.return_integral(self.z,self.dwindow[zp])
			if normalize:
				self.window[zp]=self.window[zp]/self.window_norm[zp]
				self.dwindow[zp]=self.dwindow[zp]/self.window_norm[zp]
			self.total_window=self.total_window + self.window[zp]
		
		# This is currently typically greater than unity because of the overlapping window functions
		self.gal_num_error_est=np.sum(self.nbar.values())/self.return_integral(self.z,self.dndz)
		self.total_window=self.total_window/self.gal_num_error_est
		self.gal_num_error_est_iter1=np.sum(np.array(self.nbar.values()))/self.return_integral(self.z,self.dndz)
		
		# Here we ensure that the total number of galaxies is not exceded.
		for i,zp in enumerate(self.z_centroid):
			self.nbar[zp]=self.nbar[zp]/self.gal_num_error_est_iter1
			
		self.gal_num_error_est=np.sum(np.array(self.nbar.values()))/self.return_integral(self.z,self.dndz)
	# Window function related functions
	def return_tanh_window(self,z,zmin,zmax,taper_width=0.01):
		wL = (z-zmin)/taper_width
		wS = (zmax-z)/taper_width
		W= 0.25 *(1+ np.tanh(wS)) *(1.+ np.tanh(wL))
		return W

	def return_derivative(self,x,y,eps=0.01):
		f=interp1d(x,y,kind="cubic",bounds_error=False,fill_value="extrapolate")
		fprime=(f(x+eps)-f(x-eps))/(2.*eps)
		return fprime

	def return_bias(self,z):
		bias=self.c4*np.exp(z*self.c5)
		return bias
	
	def return_dndz(self,z):
		# Without the normalization the function returns galaxy number density / deg^2
		norm=(180./np.pi)**2.
		dndz=norm*pow(10.,self.c1)*pow(z,self.c2)*np.exp(-self.c3*z)
		return dndz
	
	def setup_radial_res(self,nu0=1420.,delta_nu=12.8,z1=1.1):
		'''
		SKA2 spectroscopic resolution
		Buggy, check the units for delta_nu and nu0
		'''
		delta_nu=delta_nu*1e-3 # Assumes delta_nu is always given in kilo hertz
		self.sigma_nu = (delta_nu*10.**(3.0))/(np.sqrt(8.*np.log(2.)))
		self.sigma_z = ((1+z1)**2.)*(self.sigma_nu/(nu0*10.**(6.0)))

	def init_camb(self,cos_par={},lmax=2048,limber_phi_lmin=20):
		self.lmax=lmax ; self.ell=np.arange(0, self.lmax+1)
		self.norm_ell=np.ones(np.size(self.ell),dtype=np.float64)
		self.norm_ell[2:]=2.*np.pi/(self.ell[2:]*(self.ell[2:]+1.))
		
		self.limber_phi_lmin=limber_phi_lmin
		self.pars = camb.CAMBparams()
		if cos_par:
			self.cos_par=cos_par
		else:
			self.cos_par={}
			self.cos_par["H0"]=67.5
			self.cos_par["ombh2"]=0.022
			self.cos_par["omch2"]=0.1197
			self.cos_par["As"]=2e-9
			self.cos_par["ns"]=0.954
		
		self.pars.set_cosmology(H0=self.cos_par["H0"], ombh2=self.cos_par["ombh2"], omch2=self.cos_par["omch2"])
		self.pars.InitPower.set_params(As=self.cos_par["As"], ns=self.cos_par["ns"])
		self.pars.set_for_lmax(self.lmax, lens_potential_accuracy=1)
		#set Want_CMB to true if you also want CMB spectra or correlations
		self.pars.Want_CMB = False
		#NonLinear_both or NonLinear_lens will use non-linear corrections
		self.pars.NonLinear = model.NonLinear_both

		#The non-linear model can be changed like this:
		self.pars.set_matter_power(redshifts=[0.], kmax=2.0)
		self.pars.NonLinearModel.set_params(halofit_version='takahashi')

		self.pars.SourceTerms=model.SourceTermParams(limber_windows=True,
		limber_phi_lmin=self.limber_phi_lmin,
		counts_density=True,
		counts_redshift=True,
		counts_lensing=True,
		counts_velocity=False,
		counts_radial=False,
		counts_ISW=True,
		counts_potential=True,
		counts_evolve=False)

	def return_integral(self,x,y):
		fn=interp1d(x,y,kind="cubic",bounds_error=None,fill_value="extrapolate")
		ans,err=quad(fn,min(x),max(x))
		return ans

#	Obsolete functions
#
#	#clustering bias function, (4.2) in  1501.03990
#	def return_bias_z0ujy(self,z):
#		return 0.8695*np.exp(z*0.2338)
#
#	# Mean number of galaxies per ????
#	def return_dndz_0ujy(self,z):
#		# This needs to be appropriately normalized to estimate the poisson noise.
#		norm=((180./np.pi)**2.)*(3600.)
#		dndz = (1./norm)*pow(10,6.23)*pow(z,1.82)*np.exp(-0.98*z)
#		return dndz
