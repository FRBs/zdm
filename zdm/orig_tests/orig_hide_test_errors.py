
"""
See README.txt for an explanation
When running, look for do_something=False and do_something=True statements to turn functionalities on/off
Also look for New=True or LOAD=False statements and change these once they have been run; intermediate data
gets saved to Pickle for massive speedups (e.g. if just fine-tuning plots); but it also takes up space!

This is a legacy file from the original 'method' paper and will likely not be updated.

"""
from zdm import cosmology as cos

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
import os
import sys

import scipy as sp
from matplotlib.ticker import NullFormatter
import iteration as it
import pickle

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

##### definitions of parameters ########
	# pset defined as:
	# [0]:	log10 Emin
	# [1]:	log10 Emax
	# [2]:	alpha (spectrum: nu^alpha)
	# [3]:	gamma
	# [4]:	sfr n
	# [5}: log10 mean host DM
	# [6]: log10 sigma host DM

from misc_functions import *
from errors_misc_functions import *
import pcosmic

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
	#cos.set_cosmology(Omega_m=1.2) setup for cosmology; just in case you wish to change it! 
	cos.init_dist_measures()
	
	# get the grid of p(DM|z)
	zDMgrid, zvals,dmvals=get_zdm_grid(new=False,plot=False,method='analytic')
	# NOTE: if this is new, we also need new surveys and grids!
	
	# constants of beam method
	thresh=0
	method=2
	
	# constants of intrinsic width distribution
	Wlogmean=1.70267
	Wlogsigma=0.899148
	DMhalo=50
	
	
	#### these surveys combine samples! ####
	NewSurveys=False
	sprefix='Full'
	
	
	if sprefix=='Full':
		Wbins=10
		Wscale=2
		Nbeams=[20,20,20]
	elif sprefix=='Std':
		Wbins=5
		Wscale=3.5
		Nbeams=[5,5,10]
	# for some reason 'survey' gets over-written in previous imports. Not sure why/how. Bug?
	# For now, import it here.
	import survey
	
	if NewSurveys:
		#load the lat50 survey data
		lat50=survey.survey()
		#lat50.process_survey_file('Surveys/CRAFT_FE.dat')
		lat50.process_survey_file('Surveys/CRAFT_class_I_and_II.dat')
		lat50.init_DMEG(DMhalo)
		lat50.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
		pwidths,pprobs=survey.make_widths(lat50,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
		efficiencies=lat50.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
		
		
		# load ICS data
		ICS=survey.survey()
		ICS.process_survey_file('Surveys/CRAFT_ICS.dat')
		ICS.init_DMEG(DMhalo)
		ICS.init_beam(nbins=Nbeams[1],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
		pwidths,pprobs=survey.make_widths(ICS,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
		efficiencies=ICS.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
		
		# load Parkes data
		pks=survey.survey()
		pks.process_survey_file('Surveys/parkes_mb_class_I_and_II.dat')
		pks.init_DMEG(DMhalo)
		pks.init_beam(nbins=Nbeams[2],method=2,plot=False,thresh=thresh) # need more bins for Parkes!
		pwidths,pprobs=survey.make_widths(pks,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
		efficiencies=pks.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
		
		
		names=['ASKAP/FE','ASKAP/ICS','Parkes/Mb']
		
		surveys=[lat50,ICS,pks]
		with open('Pickle/'+sprefix+'surveys.pkl', 'wb') as output:
			pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
			pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Pickle/'+sprefix+'surveys.pkl', 'rb') as infile:
			surveys=pickle.load(infile)
			names=pickle.load(infile)
			lat50=surveys[0]
			ICS=surveys[1]
			pks=surveys[2]
	print("Initialised surveys ",names)
	dirnames=['ASKAP_FE','ASKAP_ICS','Parkes_Mb']
	
	###############################################################################
	############################### INITIALISING GRIDS ############################
	###############################################################################
	
	print("If you see errors in the following, you probably need to set 'New...=True' ")
	print("However, turn it to False once this hase been done once to save time ")
	
	
	############################ BEST n=0 #########################
	######## gets a grid of the best rates for no steller evolution #######
	n0set=np.array([30. ,  41.6 ,  1.25, -1.1  ,  0. ,   2.5 , 0.6 ,  3.68])
	NewN0=False
	
	if NewN0:
		n0grids=initialise_grids(surveys,zDMgrid, zvals,dmvals,n0set,wdist=True)
		n0rates=[]
		for g in n0grids:
			n0rates.append(g.rates)
		
		#allgrids.append(grids)
		with open('Pickle/n0rates.pkl', 'wb') as output:
			pickle.dump(n0rates, output, pickle.HIGHEST_PROTOCOL)
		#allgrids.append(grids)
		with open('Pickle/n0grids.pkl', 'wb') as output:
			pickle.dump(n0grids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Pickle/n0rates.pkl', 'rb') as infile:
			n0rates=pickle.load(infile)
		with open('Pickle/n0grids.pkl', 'rb') as infile:
			n0grids=pickle.load(infile)
	
	############################ 90% RANGES #########################
	
	NSYS=12 # 12 systematic sets
	temp = np.array([[30 , 41.6, 1.5, -0.8, 1.0, 2.0, 0.9],[
 30, 42.51, 1.5, -1.25, 1.91, 2.0, 0.45],[
30 , 41.64, 1.2, -1.12, 1.4, 2.25, 0.5],[
 30, 41.9, 1.88, -1.15, 1.9, 2.25, 0.5],[
30 , 42.08, 1.5, -1.34, 2.08, 2.25, 0.5],[
 30, 41.8, 1.5, -0.96, 1.52, 2.0, 0.6],[
30 , 41.8, 1.5, -1.1, 1.11, 2.25, 0.5],[
 30, 41.88, 1.75, -1.2, 2.28, 2.15, 0.5],[
30 , 42.18, 1.5, -1.1, 1.78, 1.77, 0.59],[
 30, 41.8, 1.5, -1.2, 1.67, 2.41, 0.56],[
30 , 42.16, 1.5, -1.2, 1.8, 2.08, 0.36],[
 30, 42.0, 1.5, -1.1, 1.6, 2.0, 0.81]])

	
	params=np.zeros([NSYS,8])
	for i in np.arange(NSYS):
		params[i,0:7]=temp[i,:]
		params[i,-1]=0.
	
	#generates zdm grids for 90% error parameter sets
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
		with open('Pickle/errorrates.pkl', 'wb') as output:
			pickle.dump(all_rates, output, pickle.HIGHEST_PROTOCOL)
		with open('Pickle/errorgrids.pkl', 'wb') as output:
			pickle.dump(all_grids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Pickle/errorrates.pkl', 'rb') as infile:
			all_rates=pickle.load(infile)
		with open('Pickle/errorgrids.pkl', 'rb') as infile:
			all_grids=pickle.load(infile)
	
	# changes all_rates to be first ordered by telescope, then by error
	
	inv_all_rates=[[],[],[]]
	for i, rset in enumerate(all_rates):
		for j,survey in enumerate(rset):
			inv_all_rates[j].append(survey)
	inv_all_grids=[[],[],[]]
	for i, rset in enumerate(all_grids):
		for j,survey in enumerate(rset):
			inv_all_grids[j].append(survey)	
	
	
	###################### BEST GRIDS #################
	
	# gets best grids (best fit)
	bestset=[30. ,  41.84 ,  1.54,  -1.16 ,  1.77 ,  2.16 , 0.51,   4.19]
	NewBest=False
	if NewBest:
		bestgrids=initialise_grids(surveys,zDMgrid, zvals,dmvals,bestset,wdist=True)
		bestrates=[]
		for g in bestgrids:
			bestrates.append(g.rates)
		
		#allgrids.append(grids)
		with open('Pickle/bestrates.pkl', 'wb') as output:
			pickle.dump(bestrates, output, pickle.HIGHEST_PROTOCOL)
		with open('Pickle/bestgrids.pkl', 'wb') as output:
			pickle.dump(bestgrids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Pickle/bestrates.pkl', 'rb') as infile:
			bestrates=pickle.load(infile)
		with open('Pickle/bestgrids.pkl', 'rb') as infile:
			bestgrids=pickle.load(infile)
	
	print("Initialised grids")
	
	############### adds minimum energy at 90% upper limit ########
	# gets best grids #
	Eminset=[38.5 ,  41.84 ,  1.54,  -1.16 ,  1.77 ,  2.16 , 0.51,   4.19]
	NewEmin=False
	if NewEmin:
		Emingrids=initialise_grids(surveys,zDMgrid, zvals,dmvals,Eminset,wdist=True)
		Eminrates=[]
		for g in Emingrids:
			Eminrates.append(g.rates)
		
		
		ConstCheck=True
		if ConstCheck:
			newC,llC,lltot=it.minimise_const_only(Eminset,Emingrids,surveys)
			#Eminset[-1]=newC
			
		#allgrids.append(grids)
		with open('Pickle/Eminrates.pkl', 'wb') as output:
			pickle.dump(Eminrates, output, pickle.HIGHEST_PROTOCOL)
		with open('Pickle/Emingrids.pkl', 'wb') as output:
			pickle.dump(Emingrids, output, pickle.HIGHEST_PROTOCOL)
	else:
		with open('Pickle/Eminrates.pkl', 'rb') as infile:
			Eminrates=pickle.load(infile)
		with open('Pickle/Emingrids.pkl', 'rb') as infile:
			Emingrids=pickle.load(infile)
	#with open('Pickle/tempgrids.pkl', 'rb') as infile:
	#		bestgrids=pickle.load(infile)
	print("Initialised Emin grid")
	
	###############################################################################
	###################### PLOTTING AND OTHER CALCULATIONS ########################
	###############################################################################
	
	#### gets the constants - initially constant were not copied over for these sets ####
	getconst=False
	if getconst:
		C,llC,lltot=it.minimise_const_only(bestset,bestgrids,surveys)
		print("best ",C)
		C,llC,lltot=it.minimise_const_only(n0set,n0grids,surveys)
		print("n0 ",C)
		C,llC,lltot=it.minimise_const_only(Eminset,Emingrids,surveys)
		print("Emin ",C)
		for i,gset in enumerate(all_grids):
			C,llC,lltot=it.minimise_const_only(params[i],gset,surveys)
			print("sys ",i,C)
	
	# does simple calculations on fractions less than 0.1 and 0.5 in z
	calc_czs=False
	if calc_czs:
		OK1=np.where(zvals < 0.1)
		OK2=np.where(zvals < 0.5)
		r1s=[]
		r2s=[]
		for g in bestrates:
			zproj=np.sum(g,axis=1)
			total=np.sum(zproj)
			total1=np.sum(zproj[OK1])
			total2=np.sum(zproj[OK2])
			r1s.append(total1/total)
			r2s.append(total2/total)
		print("STD: Within z=0.1, we have ",r1s,';    within z=0.5, we have ',r2s,'\n\n')
		for gset in all_rates:
			r1s=[]
			r2s=[]
			for g in gset:
				zproj=np.sum(g,axis=1)
				total=np.sum(zproj)
				total1=np.sum(zproj[OK1])
				total2=np.sum(zproj[OK2])
				r1s.append(total1/total)
				r2s.append(total2/total)
			print("Within z=0.1, we have ",r1s,';    within z=0.5, we have ',r2s)
	
	plot_psnrs=False
	if plot_psnrs:
		### generates snr plots ###
		gridsets=[bestgrids,n0grids,Emingrids]
		labels=['best','n0','Emin']
		psets=[bestset,n0set,Eminset]
		error_plot_psnrs(gridsets,labels,surveys,psets,plot='ErrorPlots/PSNR/snr_hist_comparison.pdf')
	
	DoEverything=False
	# this generates many plots produced in 'test.py' for *every* systematic set here.
	if DoEverything:
		for i,gridset in enumerate(all_grids):
			muDM=10**params[i][5]
			Location='ErrorPlots/'
			make_dm_redshift(gridset[0],Location+'Macquart/lat50/sys_'+str(i)+'_lat50_macquart_relation.pdf',DMmax=1000,zmax=0.75,loc='upper right',Macquart=muDM)
			make_dm_redshift(gridset[1],Location+'Macquart/ICS/sys_'+str(i)+'_ICS_macquart_relation.pdf',DMmax=2000,zmax=1,loc='upper right',Macquart=muDM)
			make_dm_redshift(gridset[2],Location+'Macquart/Pks/sys_'+str(i)+'_pks_macquart_relation.pdf',DMmax=4000,zmax=3,loc='upper left',Macquart=muDM)
			
			gpks=gridset[2]
			gICS=gridset[1]
			glat50=gridset[0]
			muDM=10**params[i,5]
			Macquart=muDM
			Location='ErrorPlots/DMZ/'
			prefix='sys_'+str(i)+'_'
			plot_grid_2(gpks.rates,gpks.zvals,gpks.dmvals,zmax=3,DMmax=3000,name=Location+'Pks/'+prefix+'nop_pks_optimised_grid.pdf',norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=pks.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
			plot_grid_2(gICS.rates,gICS.zvals,gICS.dmvals,zmax=1,DMmax=2000,name=Location+'ICS/'+prefix+'nop_ICS_optimised_grid.pdf',norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=ICS.DMEGs,FRBZ=ICS.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
			plot_grid_2(glat50.rates,glat50.zvals,glat50.dmvals,zmax=0.6,DMmax=1500,name=Location+'lat50/'+prefix+'nop_lat50_optimised_grid.pdf',norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=lat50.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
			
			plot_grid_2(gpks.rates,gpks.zvals,gpks.dmvals,zmax=3,DMmax=3000,name=Location+prefix+'pks_optimised_grid.pdf',norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=True,FRBDM=pks.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
			plot_grid_2(gICS.rates,gICS.zvals,gICS.dmvals,zmax=1,DMmax=2000,name=Location+prefix+'ICS_optimised_grid.pdf',norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=True,FRBDM=ICS.DMEGs,FRBZ=ICS.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
			plot_grid_2(glat50.rates,glat50.zvals,glat50.dmvals,zmax=0.5,DMmax=1000,name=Location+prefix+'lat50_optimised_grid.pdf',norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=True,FRBDM=lat50.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
	
	
	# need to get ks tests found and done
	compare_z_dm_fits=False
	if compare_z_dm_fits:
		### all these generate results to go into the paper ###
		compare_z_fits(surveys,bestrates,all_rates,n0rates,Eminrates,zvals,dmvals,outdir='ErrorPlots',ks=True)
		compare_dm_fits(surveys,bestrates,all_rates,n0rates,Eminrates,zvals,dmvals,outdir='ErrorPlots',ks=True)
		
		for i in np.arange(3):
			use_this=[]
			print("Doing z, DM dists for survey ",i)
			for item in all_rates:
				use_this.append([item[i]])
			if i==0:
				compare_dm_fits([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='ErrorPlots/FE')
				compare_z_fits2([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='ErrorPlots/FE',xmax=0.8,ymax=20)
			elif i==1:
				compare_dm_fits([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='ErrorPlots/ICS')
				compare_z_fits2([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='ErrorPlots/ICS',xmax=1,ymax=4)
			elif i==2:
				compare_dm_fits([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='ErrorPlots/Pks')
				compare_z_fits2([surveys[i]],[bestrates[i]],use_this,[n0rates[i]],[Eminrates[i]],zvals,dmvals,outdir='ErrorPlots/Pks',xmax=2,ymax=6)
	
	
	doSourceCounts=False # (source counts figures)
	if doSourceCounts:
		error_get_source_counts(bestgrids[0],inv_all_grids[0],Emingrids[0],plot='ErrorPlots/lat50_source_counts.pdf',Slabel='ASKAP/FE',load=True,tag='lat50')
		error_get_source_counts(bestgrids[2],inv_all_grids[2],Emingrids[2],plot='ErrorPlots/Parkes_source_counts.pdf',Slabel='Parkes/Mb',load=True,tag='Pks')	
	
	
	# not in paper - just does this for different surveys
	# Hence no guarantee the outputs are sensible
	plot_z_given_dm_priors=False
	if plot_z_given_dm_priors:
		for i,s in enumerate(surveys):
			basename='p_z_g_dm_'+dirnames[i]+'.pdf'
			dirname='ErrorPlots/'+dirnames[i]+'/'
			temp=[]
			for j,r in enumerate(all_rates):
				temp.append(r[i])
			error_get_zgdm_priors(surveys[i],bestrates[i],temp,Eminrates[i],dirname,basename,dmvals,zvals)
			
	
	
	######## calculate SNR #######
	# this was a test! Does a plot for every FRB individually
	individual_psnr=False
	if individual_psnr:
		calc_psnr_1D(Emingrids[0],surveys[0],Eminset,doplot='ErrorPlots/Emin_psnr_per_frb.pdf')
		calc_psnr_1D(bestgrids[0],surveys[0],bestset,doplot='ErrorPlots/psnr_per_frb.pdf')


main()
