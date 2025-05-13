"""
Produces Figure A7 - or at least a statistically identical version of it
"""


from zdm import io
from zdm.MC_sample import loading
import os
import numpy as np
from zdm import survey
from zdm import iteration as it
from matplotlib import pyplot as plt
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    print("Warning: produces a statistically similar plot to Figure A7")
    print("Original MC data was lost and has had to be regenerated")
    extract_CRACO_data(overwrite=True)
    make_grids(overwrite=True)
    plot_results()


def extract_CRACO_data(opdir="FigureA7/",overwrite=False):
    """
    Runs a bash script to generate modified CRACO FRB files
    """
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    # this governs which set of 1000 FRBs to use from this file
    # it can range from 0 to 4
    # using different versions produce different versions of Figure A7
    THOUSAND="4" 
    command="./modify_figA7_data.sh " + THOUSAND
    
    if (overwrite==False
        and os.path.exists(opdir+"CRACO_std_May2022_maxdm.dat")
        and os.path.exists(opdir+"CRACO_std_May2022_missing.dat") ):
        print("Modified survey files exist, skipping...")
        return 0
    
    os.system(command)

def make_grids(opdir="FigureA7/",overwrite=False):
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    ############## Load up ##############
    input_dict=io.process_jfile('../Analysis/CRACO/Cubes/craco_alpha_Emax_state.json')

    # Deconstruct the input_dict
    #state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    # force the upper incomplete gamma function spline interpolation
    #vparam_dict['energy']['luminosity_function'] = 2
    sdir="FigureA7/"
    surveys = []
    names = ['CRACO_std_May2022','CRACO_std_May2022_maxdm','CRACO_std_May2022_missing']
    savefile=opdir+"missing_ll_check.npy"
    H0file=opdir+"H0list.npy"
    if os.path.exists(savefile) and os.path.exists(H0file) and overwrite==False:
        print("Output already exists, skipping to plotting stage...")
        return None
    
    ############## Initialise survey and grids ##############
    #NOTE: grid will be identical for all three, only bother to update one!
    surveys=[]
    s,grid = loading.survey_and_grid(lum_func=2,
        survey_name=names[0],NFRB=1000,sdir=sdir) #, NFRB=Nsamples
    surveys.append(s)
    
    #sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    for i,survey_name in enumerate(names):
        if i==0:
            continue
        surveys.append(survey.load_survey(survey_name, grid.state, grid.dmvals,sdir))
    
    nH0=49
    lls=np.zeros([3,nH0])
    
    H0s=np.linspace(62,74,nH0)
    
    trueH0 = grid.state.cosmo.H0
    
    for ih,H0 in enumerate(H0s):
        
        vparams = {}
        vparams['H0'] = H0
        grid.update(vparams)
        
        lC,llC=it.minimise_const_only(
            vparams,[grid],[surveys[0]], Verbose=False,
            use_prev_grid=False)
    
        vparams = {}
        vparams['lC'] = lC
        grid.update(vparams)
        
        
        for i,s in enumerate(surveys):
            
            if s.nD==1:
                llsum=it.calc_likelihoods_1D(grid,s,psnr=True,dolist=0,Pn=True)
            elif s.nD==2:
                llsum=it.calc_likelihoods_2D(grid,s,psnr=True,dolist=0,Pn=True)
            elif s.nD==3:
                # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                llsum1=it.calc_likelihoods_1D(grid,s,psnr=True,dolist=0,Pn=True)
                llsum2=it.calc_likelihoods_2D(grid,s,psnr=True,dolist=0,Pn=False)
                llsum = llsum1+llsum2
            lls[i,ih] = llsum
            
    np.save(savefile,lls)
    np.save(H0file,H0s)

def plot_results(opdir="FigureA7/"):
    
    lls=np.load(opdir+"missing_ll_check.npy")
    savefile=opdir+"FigureA7.pdf"
    
    H0s=np.load(opdir+"H0list.npy")
    
    plt.figure()
    plt.xlabel('$H_0$ [km/s/Mpc]')
    plt.ylabel('$\\log_{10} \\ell(H_0)-\\log_{10} \\ell_{\\rm max}$')
    plt.ylim(-5,0)
    plt.xlim(62,74)
    labels=['All localised','${\\rm DM}_{\\rm EG}^{\\rm max}=1000$','$p({\\rm no}\, z)=\\frac{1}{3}$']
    styles=['-','--',':']
    for i in np.arange(3):
        peak=np.nanmax(lls[i,:])
        plt.plot(H0s[:],lls[i,:]-peak,label=labels[i],linestyle=styles[i],linewidth=3)
    trueH0=67.66
    plt.plot([trueH0,trueH0],[-5,0],linestyle='--',color='grey',label='True $H_0$')
    plt.legend(loc=[0.38,0.01])
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

#use 0 for power law, 1 for gamma function
# have not implemented Xavier's version for gamma=0 yet...
main()
