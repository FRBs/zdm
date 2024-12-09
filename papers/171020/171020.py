
from zdm import cosmology as cos
import argparse
import survey
import numpy as np

from zdm import pcosmic
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
import os
import sys

import scipy as sp

#import ne2001
#import frb
#from frb import igm
#from frb import dlas
import time
from matplotlib.ticker import NullFormatter
import iteration as it

import beams

import pickle

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

#### we now test some simple update steps ######
	# pset defined as:
	# [0]:	log10 Emin
	# [1]:	log10 Emax
	# [2]:	alpha (spectrum: nu^alpha)
	# [3]:	gamma
	# [4]:	sfr n
	# [5:}  parameters passed to dm function
from zdm.misc_functions import *
from errors_misc_functions import *
import pcosmic

#import igm
defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
	zs=[]
	### loads 171020 info ###
	with open('Data/candidate_zs.dat', 'r') as infile:
		for line in infile:
			info=line.split()
			z=float(info[4])
			zs.append(z)
	candidate_zs=np.array(zs)
	
	
	#cos.set_cosmology(Omega_m=1.2) setup for cosmology
	cos.init_dist_measures(this_ZMIN=0,this_ZMAX=0.2,this_NZ=1000)
	
	#parser.add_argument(", help
	# get the grid of p(DM|z)
	zDMgrid, zvals,dmvals=get_zdm_grid(new=False,plot=False,method='analytic',nz=400,zmax=0.1,ndm=200,dmmax=200,datdir='171020')
	
	# NOTE: if this is new, we also need new surveys and grids!
	
	# constants of beam method
	thresh=0
	method=2
	
	# constants of intrinsic width distribution
	Wlogmean=1.70267
	Wlogsigma=0.899148
	DMhalo=50
	
	#prefix='Std'
	#prefix=''
	#Wbins=5
	#Wscale=3.5
	#Nbeams=[5,5,10]
	
	#### these surveys combine samples! ####
	NewSurveys=False
	sprefix='171020'
	
	Wbins=1
	Wscale=1.4
	Nbeams=[50,20,20]
	
	if NewSurveys:
		#load the lat50 survey data
		lat50=survey.survey()
		#lat50.process_survey_file('Surveys/CRAFT_FE.dat')
		lat50.process_survey_file('Surveys/171020.dat')
		lat50.init_DMEG(DMhalo)
		#efficiencies=lat50.get_efficiency(dmvals)
		lat50.init_beam(nbins=Nbeams[0],method=2,thresh=thresh) # tells the survey to use the beam file
		pwidths,pprobs=survey.make_widths(lat50,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
		
		# 171020 is unresolved in time
		# hence the effective true width is 0
		# thus we could hard-code it
		#pprobs=np.array([1.])
		#pwidths=np.array([0.2]) # no smearing here
		# note: this is roughly what one gets anyway with nbins=0
		efficiencies=lat50.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
		
		names=['ASKAP/FE','ASKAP/ICS','Parkes/Mb']
		
		surveys=[lat50]
		with open('171020/'+sprefix+'surveys.pkl', 'wb') as output:
			pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
			pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('171020/'+sprefix+'surveys.pkl', 'rb') as infile:
			surveys=pickle.load(infile)
			names=pickle.load(infile)
			lat50=surveys[0]
	print("Initialised surveys ",names)
	dirnames=['ASKAP_FE','ASKAP_ICS','Parkes_Mb']
	
	#### these are outdated - it's mean efficiencies vs efficiency of the mean ####
	# update this for width distribution?
	plot_paper_efficiencies=False
	if plot_paper_efficiencies:
		plot_efficiencies_paper(surveys[0],'PaperPlots/lat50_compare_efficiencies.pdf',"ASKAP Fly's Eye")
		plot_efficiencies_paper(surveys[1],'PaperPlots/ICS_compare_efficiencies.pdf',"ASKAP ICS")
		plot_efficiencies_paper(surveys[2],'PaperPlots/Parkes_compare_efficiencies.pdf',"Parkes")
	
	
	PlotEfficiencies=False
	if PlotEfficiencies:
		for i,s in enumerate(surveys):
			savename='OtherPlots/'+names[i]+'_efficiencies.pdf'
			plot_efficiencies(s,savename=savename)
	
	
	############################ BEST n=0 #########################
	######## gets a grid of the best rates for no steller evolution #######
	n0set=np.array([30. ,  41.6 ,  1.25 ,-0.9 ,  0. ,   2.25 , 0.5  , 1.66])
	NewN0=False
	#prefix='' use this for standard
	prefix='171020'
	if NewN0:
		n0grids=initialise_grids(surveys,zDMgrid, zvals,dmvals,n0set,wdist=True)
		n0rates=[]
		for g in n0grids:
			n0rates.append(g.rates)
		
		#allgrids.append(grids)
		with open('Data/'+prefix+'n0rates.pkl', 'wb') as output:
			pickle.dump(n0rates, output, pickle.HIGHEST_PROTOCOL)
		#allgrids.append(grids)
		with open('Data/'+prefix+'n0grids.pkl', 'wb') as output:
			pickle.dump(n0grids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Data/'+prefix+'n0rates.pkl', 'rb') as infile:
			n0rates=pickle.load(infile)
		with open('Data/'+prefix+'n0grids.pkl', 'rb') as infile:
			n0grids=pickle.load(infile)
	
	
	############################ 90% ranges #########################
	
	NSYS=12
	temp=np.array([[30 , 41.6, 1.5, -1.2, 2.6, 2.0, 0.5],[
 30, 42.51, 1.5, -1.2, 1.69, 2.0, 0.5],[
30 , 42.0, 1.2, -1.1, 1.48, 2.0, 0.5],[
 30, 42.0, 1.88, -1.05, 1.8, 2.0, 0.5],[
30 , 42.08, 1.5, -1.34, 2.02, 2.25, 0.5],[
 30, 41.92, 1.5, -0.96, 1.4, 2.0, 0.54],[
30 , 41.8, 1.5, -1.0, 1.11, 2.25, 0.5],[
 30, 42.0, 1.75, -1.14, 2.28, 2.0, 0.5],[
30 , 42.18, 1.5, -1.1, 1.78, 1.77, 0.59],[
 30, 41.8, 1.5, -1.1, 1.47, 2.41, 0.63],[
30 , 42.08, 1.5, -1.1, 1.6, 2.0, 0.36],[
 30, 42.0, 1.5, -1.1, 1.6, 2.0, 0.81]])
	
	params=np.zeros([NSYS,8])
	for i in np.arange(NSYS):
		params[i,0:7]=temp[i,:]
		params[i,-1]=0.
	
	#generates zdm grids for 1 sigma error parameter sets
	NewGrids=False
	
	if NewGrids:
		all_grids=[]
		all_rates=[]
		for i in np.arange(NSYS):
			pset=params[i]
			
			print("About to do pset ",pset)
			grids=initialise_grids(surveys,zDMgrid, zvals,dmvals,pset,wdist=True)
			print("Initialised grids for pset ",i)
			rates=[]
			for j,g in enumerate(grids):
				rates.append(g.rates)
			all_rates.append(rates)
		
			all_grids.append(grids)
		with open('Data/'+prefix+'errorrates.pkl', 'wb') as output:
			pickle.dump(all_rates, output, pickle.HIGHEST_PROTOCOL)
		with open('Data/'+prefix+'errorgrids.pkl', 'wb') as output:
			pickle.dump(all_grids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Data/'+prefix+'errorrates.pkl', 'rb') as infile:
			all_rates=pickle.load(infile)
		with open('Data/'+prefix+'errorgrids.pkl', 'rb') as infile:
			all_grids=pickle.load(infile)
	
	# changes all_rates to be first ordered by telescope, then byerror
	
	inv_all_rates=[[],[],[]]
	for i, rset in enumerate(all_rates):
		for j,s in enumerate(rset):
			inv_all_rates[j].append(s)
	inv_all_grids=[[],[],[]]
	for i, rset in enumerate(all_grids):
		for j,s in enumerate(rset):
			inv_all_grids[j].append(s)	
	
	
	###################### BEST GRIDS #################
	
	# gets best grids #
	bestset=np.array([30. ,  41.7  ,  1.55 , -1.09 ,  1.67 ,  2.11  ,  0.53 ,  3.15])
	NewBest=False
	if NewBest:
		bestgrids=initialise_grids(surveys,zDMgrid, zvals,dmvals,bestset,wdist=True)
		bestrates=[]
		for g in bestgrids:
			bestrates.append(g.rates)
		
		#allgrids.append(grids)
		with open('Data/'+prefix+'bestrates.pkl', 'wb') as output:
			pickle.dump(bestrates, output, pickle.HIGHEST_PROTOCOL)
		with open('Data/'+prefix+'bestgrids.pkl', 'wb') as output:
			pickle.dump(bestgrids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Data/'+prefix+'bestrates.pkl', 'rb') as infile:
			bestrates=pickle.load(infile)
		with open('Data/'+prefix+'bestgrids.pkl', 'rb') as infile:
			bestgrids=pickle.load(infile)
	#with open('Pickle/tempgrids.pkl', 'rb') as infile:
	#		bestgrids=pickle.load(infile)
	print("Initialised grids")
	
	#compare_dm_fits(surveys,bestrates,all_rates,n0rates,zvals,dmvals)
	#compare_z_fits(surveys,bestrates,all_rates,n0rates,zvals,dmvals)
	
	############### adds hypothetical Emin of 10^36, tries this out ########
	# gets best grids #
	Eminset=np.array([39.0 ,  41.7  ,  1.55,  -1.09 ,  1.67 ,  2.11  ,  0.53  , 3.15])
	NewEmin=False
	if NewEmin:
		Emingrids=initialise_grids(surveys,zDMgrid, zvals,dmvals,Eminset,wdist=True)
		Eminrates=[]
		for g in Emingrids:
			Eminrates.append(g.rates)
		
		#allgrids.append(grids)
		with open('Data/Eminrates.pkl', 'wb') as output:
			pickle.dump(Eminrates, output, pickle.HIGHEST_PROTOCOL)
		with open('Data/Emingrids.pkl', 'wb') as output:
			pickle.dump(Emingrids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Data/Eminrates.pkl', 'rb') as infile:
			Eminrates=pickle.load(infile)
		with open('Data/Emingrids.pkl', 'rb') as infile:
			Emingrids=pickle.load(infile)
	#with open('Pickle/tempgrids.pkl', 'rb') as infile:
	#		bestgrids=pickle.load(infile)
	print("Initialised Emin grid")
	
	plot_psnrs=False
	if plot_psnrs:
		### generates snr plots ###
		gridsets=[bestgrids,n0grids,Emingrids]
		labels=['best','n0','Emin']
		psets=[bestset,n0set,Eminset]
		error_plot_psnrs(gridsets,labels,surveys,psets,plot='Emin/PSNR/snr_hist_comparison.pdf')
	
	compare_z_dm_fits=False
	if compare_z_dm_fits:
		### all these generate results to go into the paper ###
		compare_z_fits(surveys,bestrates,all_rates,n0rates,Eminrates,zvals,dmvals,outdir='Emin')
		compare_dm_fits(surveys,bestrates,all_rates,n0rates,Eminrates,zvals,dmvals,outdir='Emin')
		for i in np.arange(3):
			use_this=[]
			print("Doing z, DM dists for survey ",i)
			for item in all_rates:
				use_this.append([item[i]])
			if i==0:
				compare_dm_fits([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='Emin/FE')
				compare_z_fits2([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='Emin/FE',xmax=0.8,ymax=20)
			elif i==1:
				compare_dm_fits([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='Emin/ICS')
				compare_z_fits2([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='Emin/ICS',xmax=1,ymax=4)
			elif i==2:
				compare_dm_fits([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='Emin/Pks')
				compare_z_fits2([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='Emin/Pks',xmax=2,ymax=6)
	
	
	doSNR=False # Done!
	if doSNR:
		error_get_source_counts(bestgrids[0],inv_all_grids[0],Emingrids[0],plot='Emin/lat50_source_counts.pdf',Slabel='ASKAP/FE',load=True,tag='lat50')
		error_get_source_counts(bestgrids[2],inv_all_grids[2],Emingrids[2],plot='Emin/Parkes_source_counts.pdf',Slabel='Parkes/Mb',load=True,tag='Pks')	
	
	
	plot_z_given_dm_priors=True
	# reads in redshifts of FRB candidates
	cands=np.genfromtxt('Data/P.csv', delimiter=',')
	lp=np.array(cands[:,-3],dtype='float')
	zcands=np.array(cands[1:,-4],dtype='float')
	zcands[0]=0.00867
	
	print("zcands are ")
	for z in zcands:
		print(z)
	print("pos likelihood is ")
	for l in lp:
		print(l)
	print("p(z) is ",)
	
	if plot_z_given_dm_priors:
		for i,s in enumerate(surveys):
			basename='p_z_g_dmsnr_'+dirnames[i]+'.pdf'
			#dirname='ErrorPlots/'+dirnames[i]+'/'
			dirname='Data/'
			temp=[]
			for j,r in enumerate(all_grids):
				temp.append(r[i])
			print("Evaluating on DM ",[surveys[i].DMEGs[0]])
			error_get_zgdmsnr_priors(surveys[i],bestgrids[i],temp,Emingrids[i],dirname,basename,dmvals,zvals,z_evaluate=zcands,dm_evaluate=[surveys[i].DMEGs[0]])
		exit()
		for i,s in enumerate(surveys):
			basename='p_z_g_dm_'+dirnames[i]+'.pdf'
			#dirname='ErrorPlots/'+dirnames[i]+'/'
			dirname='Data/'
			temp=[]
			for j,r in enumerate(all_rates):
				temp.append(r[i])
			error_get_zgdm_priors(surveys[i],bestrates[i],temp,Eminrates[i],dirname,basename,dmvals,zvals)
			
	exit()
	
	
	######## calculate SNR #######
	# this was a test! Does a plot for every FRB individually
	#calc_psnr_1D(Emingrids[0],surveys[0],Eminset,doplot='Emin/Emin_psnr_per_frb.pdf')
	#calc_psnr_1D(bestgrids[0],surveys[0],bestset,doplot='Emin/psnr_per_frb.pdf')
	
	# OK redo this one when we have all the others. Generate a histogram. Huzzah!
	t0=time.process_time()
	err_get_source_counts([bestgrids[0]],plot='ErrorPlots/psnr.pdf')
	t1=time.process_time()
	print("Took ",t1-t0," seconds")
	# this takes a long time
	
	

main()
