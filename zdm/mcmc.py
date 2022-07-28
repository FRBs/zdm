""" Module to perform MCMC analysis """

import numpy as np

from zdm import iteration as it

def log_prob(p0, grids, surveys, param_keys):

    norm = True
    psnr = True

    # Generate vparams from p0?
    vparams = {}
    for ss, key in enumerate(param_keys):
        vparams[key] = p0[ss]

    # Generate the constant
    C,llC = it.minimise_const_only(
                vparams, grids, surveys)
    vparams['lC']=C

    # Do the Likelihood
            # in theory we could save the following step if we have already minimised but oh well. Too annoying!
    ll=0.
    longlistsum=np.array([0.,0.,0.,0.])
    alistsum=np.array([0.,0.,0.])
    for j,survey in enumerate(surveys):
        #if clone is not None and clone[j] > 0:
        #    embed(header='1047 of it -- this wont work')
        #    grids[j].copy(grids[clone[j]])
        #else:
        grids[j].update(vparams)

        if survey.nD==1:
            lls,alist,expected,longlist = it.calc_likelihoods_1D(
                grids[j],survey,norm=norm,psnr=psnr,dolist=5)
        elif survey.nD==2:
            lls,alist,expected,longlist = it.calc_likelihoods_2D(
                grids[j],survey,norm=norm,psnr=psnr,dolist=5)
        elif survey.nD==3:
            # mixture of 1 and 2D samples. NEVER calculate Pn twice!
            llsum1,alist1,expected1,longlist1 = it.calc_likelihoods_1D(
                grids[j],survey,norm=norm,psnr=psnr,dolist=5)
            llsum2,alist2,expected2, longlist2 = it.calc_likelihoods_2D(
                grids[j],survey,norm=norm,psnr=psnr,dolist=5,Pn=False)
            lls = llsum1+llsum2
            # adds log-likelihoods for psnrs, pzdm, pn
            # however, one of these Pn *must* be zero by setting Pn=False
            #alist = [alist1[0]+alist2[0], alist1[1]+alist2[1], alist1[2]+alist2[2]] #messy!
            #expected = expected1 #expected number of FRBs ignores how many are localsied
            #longlist = [longlist1[0]+longlist2[0], longlist1[1]+longlist2[1], 
            ##            longlist1[2]+longlist2[2],
            #            longlist1[3]+longlist2[3]] #messy!
        else:
            raise ValueError("Unknown code ",survey.nD," for dimensions of survey")
                # these are slow operations but negligible in the grand scheme of things
                

    return lls

# Call to emcee might be something like:
#   sampler = emcee.EnsembleSamler(nwalkers, ndim, log_prob, args=[grids, surveys, param_keys])
#   sampler.run_mcmc(p0, 10000)