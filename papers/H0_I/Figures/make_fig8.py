"""
This is a script used to produce figures for fig 8

It generates two sets of results:
- constraints on alpha (in directory fig8_alphaSingleFigures)
- constraints on other 5 non-H0 parameters (in directory fig_othersSingleFigures)

Alpha requires special treatment due to the prior not covering
the full range of possible values.
"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt

def main(verbose=True):
    
    ######### other results ####
    Planck_H0 = 67.4
    Planck_sigma = 0.5
    Reiss_H0 = 74.03
    Reiss_sigma = 1.42
    
    # output directory
    opdir="Figure9/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    CubeFile='real_down_full_cube.npz'
    if os.path.exists(CubeFile):
        data=np.load(CubeFile)
    else:
        print("Could not file cube output file ",CubeFile)
        print("Please obtain it from [repository]")
        exit()
    
    if verbose:
        for thing in data:
            print(thing)
    
    lst = data.files
    lldata=data["ll"]
    params=data["params"]
    
    if False:
        combined1=data["pzDM"]+data["pDM"]
        combined2=data["pDMz"]+data["pz"]
        diff=(combined1-combined2)/(combined1+combined2)
    
    param_vals=ac.get_param_values(data,params)
    
    #reconstructs total pdmz given all the pieces, including unlocalised contributions
    pDMz=data["P_zDM0"]+data["P_zDM1"]+data["P_zDM2"]+data["P_zDM3"]+data["P_zDM4"]
    
    #DM only contribution - however it ignores unlocalised DMs from surveys 1-3
    pDMonly=data["pDM"]+data["P_zDM0"]+data["P_zDM4"]
    
    #do this over all surveys
    P_s=data["P_s0"]+data["P_s1"]+data["P_s2"]+data["P_s3"]+data["P_s4"]
    P_n=data["P_n0"]+data["P_n1"]+data["P_n2"]+data["P_n3"]+data["P_n4"]
    
    #labels=['p(N,s,DM,z)','P_n','P(s|DM,z)','p(DM,z)all','p(DM)all','p(z|DM)','p(DM)','p(DM|z)','p(z)']
    #for datatype in [data["ll"],P_n,P_s,pDMz,pDMonly,data["pzDM"],data["pDM"],data["pDMz"],data["pz"]]:
    
    # builds uvals list
    uvals=[]
    latexnames=[]
    for ip,param in enumerate(data["params"]):
        # switches for alpha
        if param=="alpha":
            uvals.append(data[param]*-1.)
        else:
            uvals.append(data[param])
        if param=="alpha":
            latexnames.append('$\\alpha$')
            ialpha=ip
        elif param=="lEmax":
            latexnames.append('$\\log_{10} E_{\\rm max}$')
        elif param=="H0":
            latexnames.append('$H_0$')
        elif param=="gamma":
            latexnames.append('$\\gamma$')
        elif param=="sfr_n":
            latexnames.append('$n_{\\rm sfr}$')
        elif param=="lmean":
            latexnames.append('$\\mu_{\\rm host}$')
        elif param=="lsigma":
            latexnames.append('$\\sigma_{\\rm host}$')
    
    ################ single plots, no priors ############
    deprecated,uw_vectors,wvectors=ac.get_bayesian_data(data["ll"])
    
    # This produces single plots without any fancy secondary plots.
    # Tt does not assume any priors on any parameter, other than the uniform
    # priors equal to the range of data simulated.
    # Thus it also does not adjust for undercoverage of alpha and H0 in priors,
    # hence the limits file in "limits.dat" applies only for other parameters.
    #ac.do_single_plots(uvals,uw_vectors,None,params,tag="basic_",truth=None,dolevels=True,latexnames=latexnames)
    
    ########### H0 data for fixed values of other parameters ###########
    # this generates sets of constraints on parameters with various priors
    # on H0. "vals1" will contain values assuming the Riess value of H0
    # "vals3" contains values assuming Planck priors on H0
    # "vals2" is for the best-fit values of all parameters *other*
    # than H0, i.e. will be used to get constraints on H0 when
    # everything else is held fixed
    
    # extracts best-fit values
    list1=[]
    vals1=[]
    list2=[]
    vals2=[]
    vals3=[]
    for i,vec in enumerate(uw_vectors):
        n=np.argmax(vec)
        val=uvals[i][n]
        if params[i] == "H0":
            list1.append(params[i])
            vals1.append(Reiss_H0)
            
            vals3.append(Planck_H0)
            
            iH0=i # setting index for Hubble
        else:
            list2.append(params[i])
            vals2.append(val)
    
    # gets the slice corresponding to the best-fit values of H0
    # i.e. assuming Reiss and Planck values for H0 as a prior
    Reiss_H0_selection=ac.get_slice_from_parameters(data,list1,vals1)
    Planck_H0_selection=ac.get_slice_from_parameters(data,list1,vals3)
    
    # These thus produce Bayesian limits over everything but H0
    deprecated,ReissH0_vectors,deprecated=ac.get_bayesian_data(Reiss_H0_selection)
    deprecated,PlanckH0_vectors,deprecated=ac.get_bayesian_data(Planck_H0_selection)
    
    # gets the slice corresponding to the best-fit values of all other parameters
    # this is 1D, so is our limit on H0 keeping all others fixed
    # i.e. it mimics what other works do, where H0 is varied for an
    # assumed set of other FRB parameters.
    pH0_fixed=ac.get_slice_from_parameters(data,list2,vals2)
    
    ####### 1D plots for prior on H0 ########
    
    # applies a prior on H0, which is flat between systematic differences,
    # then falls off as a Gaussian either side. This is used to generate
    # the solid blue lines in Figure 8
    H0_dim=np.where(params=="H0")[0][0]
    wlls = ac.apply_H0_prior(data["ll"],H0_dim,data["H0"],Planck_H0,
        Planck_sigma, Reiss_H0, Reiss_sigma)
    deprecated,wH0_vectors,wvectors=ac.get_bayesian_data(wlls)
    
    # these plots are not used in the paper
    # but they show constraints on parameters equal to the solid blue lines
    # in Figure 8 only, i.e. without the other grey lines
    # These may be useful for e.g. conference talks.
    # However, the H0 plot thus produced
    #ac.do_single_plots(uvals,wH0_vectors,None,params,tag="fig8_nogrey_",truth=None,
    #    dolevels=True,latexnames=latexnames,logspline=False)
    
    pH0_fixed -= np.max(pH0_fixed)
    pH0_fixed = 10**pH0_fixed
    pH0_fixed /= np.sum(pH0_fixed)
    pH0_fixed /= (uvals[iH0][1]-uvals[iH0][0])
    
    ######### combines into single plot #######
    #other results: pH0_fixed all others, and unweighted
    # this result plots H0 without any extrapolation
    #others=[[pH0_fixed]]
    #ac.do_single_plots([uvals[iH0]],[uw_vectors[iH0]],None,[params[iH0]],tag="no_extrapolation_fig6_H0",
    #    truth=None,dolevels=True,latexnames=[latexnames[iH0]],logspline=False,others=others)
    
    # now do this with others...
    # builds others...(as in, other likelihoods with other assumptions)
    others=[]
    for i,p in enumerate(params):
        if i==iH0:
            oset=None
            others.append(oset)
        else:
            if i<iH0:
                modi=i
            else:
                modi=i-1
            oset=[uw_vectors[i],ReissH0_vectors[modi],PlanckH0_vectors[modi]]
            others.append(oset)
    
    # special plot for alpha, where confidence intervals are LIES (due to undercoverage of the prior)
    do_alpha_plot([uvals[ialpha]],[wH0_vectors[ialpha]],None,[params[ialpha]],tag="fig8_alpha",truth=None,
        dolevels=True,latexnames=[latexnames[ialpha]],logspline=False,others=[others[ialpha]])
    
    # removes alpha and H0 from the list of parameters
    del uvals[ialpha]
    del wH0_vectors[ialpha]
    params=np.delete(params,ialpha)
    del latexnames[ialpha]
    del others[ialpha]
    
    if iH0 > ialpha: #will be smaller list now
        iH0 = iH0 -1
    del uvals[iH0]
    del wH0_vectors[iH0]
    params=np.delete(params,iH0)
    del latexnames[iH0]
    del others[iH0]
    
    # remaining plots
    ac.do_single_plots(uvals,wH0_vectors,None,params,tag="fig8_others",truth=None,
        dolevels=True,latexnames=latexnames,logspline=False,others=others)
    
    

def do_alpha_plot(uvals,vectors,wvectors,names,tag=None, fig_exten='.png',
                    dolevels=False,log=True,outdir='SingleFigs/',
                    vparams_dict=None, prefix='',truth=None,latexnames=None,
                    logspline=True, others=None):
    """ Generate a series of 1D plots of the cube parameters
    Does special treatment for the "alpha" parameter

    Args:
        uvals (list): List, each element containing a
            np.ndarray giving the parameter values
            for each parameter. Total length nparams.
        vectors (list): [For each parameter, contains
            an unweighted vector giving
            1D probabilities for that value of the parameter]
        wvectors ([list]): [For each parameter, contains
            a weighted (with prior) vector giving
            1D probabilities for that value of the parameter]
        names ([type]): [description]
        tag ([type], optional): [description]. Defaults to None.
        fig_exten (str, optional): [description]. Defaults to '.png'.
        dolevels (bool, optional): [description]. Defaults to False.
        log (bool, optional): [description]. Defaults to True.
        outdir (str, optional): [description]. Defaults to 'SingleFigs/'.
        vparams_dict (dict, optional): parameter dict -- used to set x-values. Defaults to None.
        prefix (str, optional): [description]. Defaults to ''.
        logspline(bool): do spline fitting in logspace?
        others(list of arrays): list of other plots to add to data

    """
    import os
    import math
    
    if tag is not None:
        outdir=tag+outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    
    if log:
        logfile=outdir+'limits.dat'
        logfile=open(logfile,'w')
    
    if dolevels:
        results=np.zeros([len(uvals),9]) # holds mean and error info for each parameter
        prior_results=np.zeros([len(uvals),9]) # does the same with alpha priors
    
    for i,vals in enumerate(uvals):
        if len(vals) == 1:
            continue
        if len(vals) < 4:
            kind = 'linear'
        else:
            kind = 'cubic'
        # does the for alpha
        plt.figure()
        lw=3
        

        # Convert vals?
        if vparams_dict is not None:
            # Check
            assert vparams_dict[names[i]]['n'] == len(vals)
            vals = np.linspace(vparams_dict[names[i]]['min'], 
                               vparams_dict[names[i]]['max'],
                               len(vals))
        
        # get raw ylimits
        # removes zeroes, could lead to strange behaviour in theory
        ymax=np.max(vectors[i])
        temp=np.where((vectors[i] > 0.) & (np.isfinite(vectors[i])) )
        
        # set to integers and get range
        ymax=math.ceil(ymax)
        ymin=0.
        
        x,y=ac.interpolate_points(vals[temp],vectors[i][temp],logspline)
        
        norm=np.sum(y)*(x[1]-x[0]) # integral y dx ~ sum y delta x
        norm=np.abs(norm)
        y /= norm
        vectors[i][temp] /= norm
        plt.plot(x,y,label='Uniform',color='blue',linewidth=lw,linestyle='-')
        plt.plot(vals[temp],vectors[i][temp],color='blue',linestyle='',marker='s')
        
        
        # weighted plotting
        if wvectors is not None:
            wx,wy=ac.interpolate_points(vals[temp],wvectors[i][temp],logspline)
            wnorm=np.sum(wy)*(x[1]-x[0])
            wnorm = np.abs(wnorm)
        
            wvectors[i][temp] /= wnorm
            wy /= wnorm
            plt.plot(x,wy,label='Gauss',color='orange',linewidth=lw,linestyle='--')
        
        ax=plt.gca()
        ax.xaxis.set_ticks_position('both')
        #ax.Xaxis.set_ticks_position('both')
        if wvectors is not None:
            ymax=np.max([np.max(wy),np.max(y)])
        else:
            ymax=np.max(y)
        
        #ymax=(np.ceil(ymax*5.))/5.
        
        
        if dolevels==True:# and i != 1:
            limvals=np.array([0.15866])
            labels=['68%']
            styles=['-']
            upper=np.max(vectors[i])
            
            besty=np.max(y)
            imax=np.argmax(y)
            xmax=x[imax]
            results[i,0]=xmax
            string=names[i]+" & {0:4.2f}".format(xmax)
            for iav,av in enumerate(limvals):
                # need to integrate from min to some point
                # gets cumulative distribution
                # sets intervals according to highest likelihood
                v0,v1,ik1,ik2=ac.extract_limits(x,y,av,method=1)
                
                v0=0.15
                v1=1.85
                ik1=np.where(x>-0.15)[0][-1]
                ik2=np.where(x<-1.85)[0][0]
                
                string += " & $_{"
                string += "{0:4.2f}".format(v0-xmax)
                string += "}^{+"
                string += "{0:4.2f}".format(v1-xmax)
                string += "}$ "
                results[i,2*iav+1]=v0-xmax
                results[i,2*iav+2]=v1-xmax
                
                hl=0.03
                doff=(x[-1]-x[0])/100.
                ybar=(av+ymax)/2.
                xbar=(v0+v1)/2.
                
                # need to separate the plots
                if wvectors is not None:
                    if ik1 != 0:
                        #if iav==3 and i==4:
                        #    ybar -= 0.8
                        plt.plot([x[ik1],x[ik1]],[ymax,y[ik1]],color='blue',linestyle=styles[iav],alpha=0.5)
                        if i==1:
                            t=plt.text(x[ik1]+doff*0.5,(ymax)+(-3.6+iav)*0.2*ymax,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                    if ik2 != wy.size-1:
                        plt.plot([x[ik2],x[ik2]],[ymax,y[ik2]],color='blue',linestyle=styles[iav],alpha=0.5)
                        if i != 1:
                            t=plt.text(x[ik2]-doff*3,(ymax)+(-3.6+iav)*0.2*ymax,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                else:
                    plt.plot([x[ik1],x[ik1]],[0,y[ik1]],color='red',linestyle=styles[iav])
                    plt.plot([x[ik2],x[ik2]],[0,y[ik2]],color='red',linestyle=styles[iav])
                    Dx=x[-1]-x[0]
                    if Dx < 0.:
                        plt.text(x[ik1],y[ik1]+ymax*0.05,labels[iav],color='red',rotation=90)
                        plt.text(x[ik2]+0.02*Dx,y[ik2]+ymax*0.05,labels[iav],color='red',rotation=90)
                    else:
                        plt.text(x[ik1]-0.02*Dx,y[ik1]+ymax*0.05,labels[iav],color='red',rotation=90)
                        plt.text(x[ik2],y[ik2]+ymax*0.05,labels[iav],color='red',rotation=90)
                    #print("For parameter ",i," CI ",iav, " is ",x[ik1]," to ",x[ik2])
            string += " & "
            
        #could just ignore the weightings   
        if wvectors is not None:
            plt.plot(vals[temp],wvectors[i][temp],color='orange',linestyle='',marker='o')
            if dolevels==True:
                limvals=np.array([0.0015,0.025,0.05,0.16])
                labels=['99.7%','95%','90%','68%']
                styles=['--',':','-.','-']
                upper=np.max(wvectors[i])
                
                besty=np.max(wy)
                imax=np.argmax(wy)
                xmax=x[imax]
                prior_results[i,0]=xmax
                string+=" {0:4.2f}".format(xmax)
                for iav,av in enumerate(limvals):
                    
                    # sets intervals according to highest likelihood
                    v0,v1,ik1,ik2=ac.extract_limits(x,wy,av,method=1)
                    
                    string += " & $_{"
                    string += "{0:4.2f}".format(v0-xmax)
                    string += "}^{+"
                    string += "{0:4.2f}".format(v1-xmax)
                    string += "}$ "
                    prior_results[i,2*iav+1]=v0-xmax
                    prior_results[i,2*iav+2]=v1-xmax
                    
                    # version 2
                    hl=0.03
                    
                    doff=(x[-1]-x[0])/100.
                    if i==1:
                        doff=0.001
                    ybar=(av+ymin)/2.
                    xbar=(v0+v1)/2.
                    if ik1 != 0:
                        plt.plot([x[ik1],x[ik1]],[ymin,wy[ik1]],color='orange',linestyle=styles[iav])
                        if i ==1:
                            t=plt.text(x[ik1]+doff*0.5,wy[ik1]/2.2,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
                        
                    if ik2 != wy.size-1:
                        
                        plt.plot([x[ik2],x[ik2]],[ymin,wy[ik2]],color='orange',linestyle=styles[iav])
                        if i != 1:
                            t=plt.text(x[ik2]-doff*3,wy[ik2]/2.2,labels[iav],rotation=90,fontsize=12)
                            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white',pad=-1))
        other_styles=[":","--","-."]
        # plot any other plots
        if others is not None:
            if others[i] is not None:
                for io,data in enumerate(others[i]):
                    x,y=ac.interpolate_points(vals,data,logspline)
                    norm=np.sum(y)*(x[1]-x[0]) # integral y dx ~ sum y delta x
                    norm=np.abs(norm)
                    y /= norm
                    plt.plot(x,y,color='grey',linewidth=1,linestyle=other_styles[io % 3])
        if dolevels:
            string += "\\\\"
            if log:
                logfile.write(string+'\n')
            else:
                print(string)
        #plt.ylim(0.,ymax)
        plt.gca().set_ylim(bottom=0)
        if truth is not None:
            plt.plot([truth[i],truth[i]],plt.gca().get_ylim(),color='black',linestyle=':')
            Dx=x[-1]-x[0]
            plt.text(truth[i]+0.01*Dx,ymax*0.4,'simulated truth',rotation=90)
        
        if latexnames is not None:
            plt.xlabel(latexnames[i])
            plt.ylabel('$p($'+latexnames[i]+'$)$')
        else:
            plt.xlabel(names[i])
            plt.ylabel('p('+names[i]+')')
        if i==4 and wvectors is not None:
            plt.legend(loc='upper left',title='Prior on $\\alpha$')
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, prefix+names[i]+fig_exten), dpi=200)
        plt.close()
    if log:
        logfile.close()
    if dolevels:
        return results,prior_results
    else:
        return 

main()
